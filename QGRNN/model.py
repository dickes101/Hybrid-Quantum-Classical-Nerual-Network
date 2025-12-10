import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from mindquantum.core.operators import QubitOperator
from mindquantum.core.operators import Hamiltonian
from mindquantum.framework import MQLayer
from mindquantum.simulator import Simulator

from .classical_layer import GraphAttentionLayer
from .fusion import GatedFusion
from .quantum_layer import build_simplified_quantum_circuit


class ImprovedHybridQGNN(nn.Cell):
    def __init__(self, in_feats, hidden_size, num_classes, num_qubits=8, quantum_depth=2):
        super(ImprovedHybridQGNN, self).__init__()
        self.gat1 = GraphAttentionLayer(in_feats, hidden_size)
        self.gat2 = GraphAttentionLayer(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(0.5)

        self.num_qubits = num_qubits
        self.to_quantum = nn.Dense(hidden_size // 2, num_qubits)
        self.fusion = GatedFusion(hidden_size // 2, num_qubits, hidden_size // 2)
        self.classifier = nn.Dense(hidden_size // 2, num_classes)
        
        self.circuit = build_simplified_quantum_circuit(num_qubits, quantum_depth)
        self._init_quantum_layer()
            
    def _init_quantum_layer(self):
        hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in range(self.num_qubits)]
        sim = Simulator('mqvector', self.circuit.n_qubits)
        grad_ops = sim.get_expectation_with_grad(hams, self.circuit)
        self.quantum_layer = MQLayer(grad_ops, weight='normal')

    def construct(self, x, adj, vq_indices=None, return_quantum=False):
        h = self.gat1(x, adj)
        h = nn.LeakyReLU(0.2)(h)
        h = self.dropout(h)
        h = self.gat2(h, adj)
        h = nn.LeakyReLU(0.2)(h)
        classical_features = h

        quantum_input = self.to_quantum(h)
        quantum_input = ops.tanh(quantum_input) * np.pi
        quantum_out = self.quantum_layer(quantum_input)

        if vq_indices is not None and vq_indices.size > 0:
            fused = classical_features.copy()
            fused[vq_indices] = self.fusion(classical_features[vq_indices], quantum_out[vq_indices])
        else:
            fused = self.fusion(classical_features, quantum_out)

        output = self.classifier(fused)
        return output, h