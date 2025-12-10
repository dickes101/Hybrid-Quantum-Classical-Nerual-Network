from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RY, RZ, RX, CNOT, H, X, Z, U3


def build_simplified_quantum_circuit(num_qubits: int, depth: int):

    enc = Circuit()
    for q in range(num_qubits):
        enc += H.on(q)
        enc += RY(f'enc_{q}').on(q)

    ansatz = Circuit()
    for d in range(depth):
        for q in range(num_qubits):
            ansatz += RY(f'ry_{d}_{q}').on(q)
            ansatz += RZ(f'rz_{d}_{q}').on(q)
            
    ring = Circuit()
    for q in range(num_qubits - 1):
        ring += CNOT.on(q + 1, q)
    ring += CNOT.on(0, num_qubits - 1)

    return enc.as_encoder() + ansatz.as_ansatz() + ring

# Circuit a
def build_simplified_quantum_circuit_a(num_qubits: int, depth: int):

    enc = Circuit()
    for q in range(num_qubits):
        enc += H.on(q)
        enc += RY(f'enc_{q}').on(q)

    ansatz = Circuit()
    for q in range(num_qubits):
        ansatz += U3(
            f'u3_theta_{q}',
            f'u3_phi_{q}',
            f'u3_lambda_{q}'
        ).on(q)
    for d in range(depth):
        for q in range(0, num_qubits, 2):
            ansatz += CNOT.on(q + 1, q)
        for q in range(0, num_qubits, 2):
            ansatz += RY(f'ry1_{d}_{q}').on(q)
        for q in range(1, num_qubits, 2):
            ansatz += RZ(f'rz1_{d}_{q}').on(q)
        for q in range(0, num_qubits, 2):
            ansatz += CNOT.on(q, q + 1)
        for q in range(0, num_qubits, 2):
            ansatz += RY(f'ry1_{d}_{q}').on(q)
        for q in range(0, num_qubits, 2):
            ansatz += CNOT.on(q + 1, q)
        for q in range(num_qubits):
            ansatz += RZ(f'rz1_{d}_{q}').on(q)
            
    return enc.as_encoder() + ansatz.as_ansatz()

# Circuit b
def build_simplified_quantum_circuit_b(num_qubits: int, depth: int):

    enc = Circuit()
    for q in range(num_qubits):
        enc += H.on(q)
        enc += RY(f'enc_{q}').on(q)

    ansatz = Circuit()
    for d in range(depth):
        for q in range(num_qubits):
            ansatz += RZ(f'rz1_{d}_{q}').on(q)
        for q in range(num_qubits):
            ansatz += RY(f'ry1_{d}_{q}').on(q)
        for q in range(0, num_qubits, 2):
            if q + 1 < num_qubits:
                ansatz += CNOT.on(q + 1, q)
        for q in range(num_qubits):
            ansatz += RZ(f'rz2_{d}_{q}').on(q)

    ring = Circuit()
    for q in range(num_qubits - 1):
        ring += CNOT.on(q, q + 1)
    ring += CNOT.on(num_qubits - 1, 0)

    return enc.as_encoder() + ansatz.as_ansatz() + ring

# Circuit c
def build_simplified_quantum_circuit_c(num_qubits: int, depth: int):
    
    enc = Circuit()
    for i in range(num_qubits):
        enc += H.on(i)
        enc += RY(f'enc_{i}').on(i)

    ansatz = Circuit()
    for d in range(depth):
        for i in range(num_qubits):
            ansatz += RZ(f'rz1_{d}_{i}').on(i)
        for i in range(num_qubits - 1):
            ansatz += CNOT.on(i + 1, i)
    for d in range(depth):
        for i in range(num_qubits):
            ansatz += RY(f'ry1_{d}_{i}').on(i)
            ansatz += RZ(f'rz2_{d}_{i}').on(i)
        for i in range(num_qubits - 1):
            ansatz += CNOT.on(i + 1, i)

    return enc.as_encoder() + ansatz.as_ansatz()

# Circuit d
def build_simplified_quantum_circuit_d(num_qubits: int, depth: int):
    
    enc = Circuit()
    for q in range(num_qubits):
        enc += H.on(q)
        enc += RY(f'enc_{q}').on(q)

    ansatz = Circuit()
    for q in range(num_qubits):
        ansatz += U3(
            f'u3_theta_{q}',
            f'u3_phi_{q}',
            f'u3_lambda_{q}'
        ).on(q)
    for d in range(depth):
        ansatz += CNOT.on(1, 5)
        ansatz += CNOT.on(0, 7)
        ansatz += CNOT.on(2, 4)
        ansatz += CNOT.on(3, 6)
        for q in range(num_qubits):
            ansatz += RY(f'ry1_{d}_{q}').on(q)
        for q in range(num_qubits):
            ansatz += RZ(f'rz1_{d}_{q}').on(q)
        ansatz += CNOT.on(3, 7)
        ansatz += CNOT.on(0, 4)
        ansatz += CNOT.on(5, 2)
        ansatz += CNOT.on(1, 6)

    return enc.as_encoder() + ansatz.as_ansatz()

if __name__ == "__main__":
    circ = build_simplified_quantum_circuit(8, 1)
    print(circ.summary())
    print(circ)