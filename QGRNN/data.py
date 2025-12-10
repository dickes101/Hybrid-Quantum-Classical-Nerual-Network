import numpy as np
import mindspore as ms
from mindspore import Tensor
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree, k_hop_subgraph, to_undirected

class Dataset:
    def __init__(self, dataset='Cora', k=2, extra_num=None):
        self.dataset = dataset
        self.k = k
        self.extra_num = extra_num
        self._load_data()
        self._compute_vq_indices()

    def _load_data(self):
        dataset = Planetoid(root='./data', name=self.dataset)
        data = dataset[0]
        num_nodes = data.num_nodes

        edge_index_torch = to_undirected(data.edge_index, num_nodes=num_nodes)
        self.edge_index_torch = edge_index_torch
        edge_index_np = edge_index_torch.numpy().astype(np.int32)
        self.edge_index = edge_index_np

        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        adj[edge_index_np[0], edge_index_np[1]] = 1.0
        adj = np.maximum(adj, adj.T) 
        adj += np.eye(num_nodes, dtype=np.float32)

        rowsum = adj.sum(axis=1).astype(np.float32)
        d_inv_sqrt = np.power(rowsum, -0.5, dtype=np.float32)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        adj = adj * d_inv_sqrt[:, np.newaxis] * d_inv_sqrt[np.newaxis, :]

        self.adj = Tensor(adj, ms.float32)
        feat = data.x.numpy().astype(np.float32)
        rowsum = feat.sum(axis=1, keepdims=True)
        rowsum[rowsum == 0.0] = 1.0
        feat = feat / rowsum
        norm = np.linalg.norm(feat, axis=1, keepdims=True)
        norm[norm == 0.0] = 1.0
        feat = feat / norm

        self.features = Tensor(feat, ms.float32)
        self.labels = Tensor(data.y.numpy().astype(np.int32), ms.int32)

        train_mask = data.train_mask.numpy()
        val_mask = data.val_mask.numpy()
        test_mask = data.test_mask.numpy()
        split_info = "official Planetoid split"

        if self.extra_num is not None and self.extra_num > 0:
            rng = np.random.RandomState(42)  
            assigned_mask = train_mask | val_mask | test_mask
            unassigned = np.where(~assigned_mask)[0]
            num_extra = min(self.extra_num, len(unassigned))
            
            if num_extra > 0:
                extra_nodes = rng.choice(unassigned, size=num_extra, replace=False)
                train_mask[extra_nodes] = True

        self.train_mask = Tensor(train_mask.astype(np.float32), ms.bool_)
        self.val_mask = Tensor(val_mask.astype(np.float32), ms.bool_)
        self.test_mask = Tensor(test_mask.astype(np.float32), ms.bool_)

        print(f"Dataset: {self.dataset} | {split_info}")
        print(f"Total nodes: {num_nodes}")
        print(f"Train nodes: {int(train_mask.sum())}")
        print(f"Value nodes: {int(val_mask.sum())}")
        print(f"Test nodes: {int(test_mask.sum())}")
        print("-" * 60)

    def _compute_vq_indices(self):
        edge_index = self.edge_index_torch
        num_nodes = self.features.shape[0]
        deg = degree(edge_index[0], num_nodes=int(num_nodes))
        center_node = int(torch.argmax(deg).item())

        nodes_within_k, _, _, _ = k_hop_subgraph(center_node, self.k, edge_index, relabel_nodes=False)
        mask_outside = torch.ones(int(num_nodes), dtype=torch.bool)
        mask_outside[nodes_within_k] = False
        nodes_outside = torch.arange(int(num_nodes))[mask_outside]

        self.center_node = center_node
        self.center_degree = float(deg[center_node].item())
        self.vq_indices_np = nodes_outside.numpy().astype(np.int32)
        self.vq_indices_ms = Tensor(self.vq_indices_np, ms.int32)
        print(f"[V_Q] k={self.k} | central nodes: {center_node} (k={self.center_degree:.0f}) | nodes out of k-hop: {len(self.vq_indices_np)} / {int(num_nodes)}")