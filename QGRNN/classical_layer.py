import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class GraphAttentionLayer(nn.Cell):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super().__init__()
        self.W = nn.Dense(in_features, out_features, has_bias=False)
        self.a = nn.Dense(2 * out_features, 1, has_bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.feat_drop = nn.Dropout(1 - dropout) 
        self.attn_drop = nn.Dropout(1 - dropout)   

    def construct(self, h, adj):
        edge_index = ops.nonzero(adj > 0)         
        row = edge_index[:, 0].astype(ms.int32)    
        col = edge_index[:, 1].astype(ms.int32)    
        N = h.shape[0]
        Wh = self.W(self.feat_drop(h))    
        Wh_i = Wh[row]                               
        Wh_j = Wh[col]

        e_ij = self.leakyrelu(self.a(ops.concat((Wh_i, Wh_j), axis=1)).squeeze(1)) 
        seg_max = ops.unsorted_segment_max(e_ij, row, N)       
        e_norm = e_ij - seg_max[row]
        exp_e = ops.exp(e_norm)

        seg_sum = ops.unsorted_segment_sum(exp_e, row, N)        
        attn = exp_e / (seg_sum[row] + 1e-16)                    
        attn = self.attn_drop(attn)
        out = ops.unsorted_segment_sum(ops.expand_dims(attn, 1) * Wh_j, row, N) 
        return out