# QGRNN: Feature Fusion-Based Hybrid Quantum-Classical Graph Residual Neural Network

<img src="model.png" width="1100">

This repository provides the reference implementation of QGRNN, a hybrid quantum–classical graph residual neural network designed to enhance node representation learning through targeted quantum augmentation.

QGRNN consists of four core components:
- Structure-driven node selection, identifying non-central (marginal) nodes for quantum enhancement
- A GAT-based classical encoder, extracting structural and semantic features
- A parametrized quantum circuit (PQC) providing nonlinear expressive transformations
- A gated residual fusion mechanism, adaptively integrating classical and quantum embeddings.

The codebase includes the complete architectural design and training pipeline of QGRNN.
For illustration and reproducibility, we provide a fully runnable example on the Cora node classification task.
