import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class TrainWrapper(nn.Cell):
    def __init__(self, net, loss_fn, edge_index, lambda_reg=0.01):
        super().__init__()
        self.net = net
        self.loss_fn = loss_fn
        self.edge_index = edge_index
        self.lambda_reg = lambda_reg

    def construct(self, features, adj, labels, mask, vq_indices=None):
        logits, hidden = self.net(features, adj, vq_indices=vq_indices)
        cls_loss = self.loss_fn(logits[mask], labels[mask])
        edge_loss = 0
        if self.lambda_reg > 0:
            row, col = self.edge_index
            diff = hidden[row] - hidden[col]
            edge_loss = ops.reduce_mean(ops.reduce_sum(diff * diff, axis=1))
        else:
            edge_loss = Tensor(0.0, ms.float32)
        total_loss = cls_loss + self.lambda_reg * edge_loss
        return total_loss, cls_loss, edge_loss