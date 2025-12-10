import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class GatedFusion(nn.Cell):
    def __init__(self, classical_dim, quantum_dim, output_dim):
        super(GatedFusion, self).__init__()
        self.gate = nn.Dense(classical_dim + quantum_dim, output_dim)
        self.gate.bias.set_data(self.gate.bias + Tensor(np.ones((output_dim,), dtype=np.float32), ms.float32))
        self.transform = nn.Dense(classical_dim + quantum_dim, output_dim)
        
    def construct(self, classical, quantum):
        combined = ops.concat([classical, quantum], axis=1)
        gate = ops.sigmoid(self.gate(combined))
        transformed = ops.tanh(self.transform(combined))
        return classical + gate * (transformed - classical)