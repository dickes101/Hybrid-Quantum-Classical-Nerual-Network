# QGRNN: Feature Fusion-Based Hybrid Quantum-Classical Graph Residual Neural Network

This repository provides the reference implementation of QGRNN, a hybrid quantum–classical graph residual neural network designed to enhance node representation learning through targeted quantum augmentation.

QGRNN consists of four core components:
- Structure-driven node selection, identifying non-central (marginal) nodes for quantum enhancement
- A GAT-based classical encoder, extracting structural and semantic features
- A parametrized quantum circuit (PQC) providing nonlinear expressive transformations
- A gated residual fusion mechanism, adaptively integrating classical and quantum embeddings.
