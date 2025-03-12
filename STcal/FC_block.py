# import torch
# import torch.nn as nn
# from lib.nn.hierarchical.models.higp_tts_model import HiGPTTSModel
#
# class GraphLearn(nn.Module):
#     """
#     通过神经元特征计算图结构: (B, T, V, F) -> (B, T, V, V)
#     """
#
#     def __init__(self, alpha=0.1, device='cpu'):
#         super(GraphLearn, self).__init__()
#         self.alpha = alpha
#         self.device = device
#         self.built = False
#
#         # 存储损失
#         self.diff_loss = 0.0
#         self.f_norm_loss = 0.0
#
#     def build(self, F):
#         """ 初始化权重 """
#         self.a = nn.Parameter(torch.randn(F, 1).to(self.device))
#         self.built = True
#
#     def forward(self, x):
#         """
#         x: (B, T, V, F) - 输入神经元数据
#         return: S (B, T, V, V) - 学习到的图结构
#         """
#         B, T, V, F = x.shape
#         if not self.built:
#             self.build(F)
#
#         S_list = []
#         diff_loss_batch = 0.0
#
#         for t in range(T):
#             xt = x[:, t, :, :]  # (B, V, F)
#             x1 = xt.unsqueeze(2)  # (B, V, 1, F)
#             x2 = xt.unsqueeze(1)  # (B, 1, V, F)
#             diff = x1 - x2  # (B, V, V, F)
#
#             w_4d = torch.einsum('b i j f, f c-> b i j c', torch.abs(diff), self.a)  # (B, V, V, 1)
#             w_3d = w_4d.squeeze(-1)  # (B, V, V)
#             tmpS = torch.exp(w_3d)
#             row_sum = tmpS.sum(dim=-1, keepdim=True) + 1e-9
#             S_ = tmpS / row_sum  # (B, V, V)
#             S_list.append(S_)
#
#             # 计算 diff^2 * S_
#             diff_sq = diff.pow(2).sum(dim=-1)  # (B, V, V)
#             local_loss = (diff_sq * S_).sum(dim=[1, 2])  # (B,)
#             diff_loss_batch += local_loss.mean()  # 累加每个 t
#
#         # 堆叠得到最终的 (B, T, V, V)
#         S = torch.stack(S_list, dim=1)
#         print("S shape:", S.shape)  # 确保 S 正常
#
#         # 记录损失
#         self.diff_loss = diff_loss_batch
#         self.f_norm_loss = self.alpha * (S.pow(2).sum())
#         print("finish here")
#
#         return S
#
#
# def compute_edge_index(S):
#     """
#     计算 edge_index 和 edge_weight
#     S: (B, T, V, V) - 归一化邻接矩阵
#     return: edge_index (B, T, 2, num_edges), edge_weight (B, T, num_edges)
#     """
#     B, T, V, V = S.shape
#     edge_index_list = []
#     edge_weight_list = []
#
#     for b in range(B):  # 遍历 batch
#         edge_index_t = []
#         edge_weight_t = []
#
#         for t in range(T):  # 遍历时间步
#             adj_matrix = S[b, t]  # (V, V)
#             edge_indices = torch.nonzero(adj_matrix, as_tuple=False).T  # (2, num_edges)
#
#             # 避免 `torch.stack()` 失败
#             if edge_indices.numel() == 0:
#                 edge_indices = torch.empty((2, 0), dtype=torch.long, device=S.device)
#                 edge_weights = torch.empty((0,), dtype=torch.float32, device=S.device)
#             else:
#                 edge_weights = adj_matrix[edge_indices[0], edge_indices[1]]  # (num_edges,)
#
#             edge_index_t.append(edge_indices)
#             edge_weight_t.append(edge_weights)
#
#         edge_index_list.append(edge_index_t)
#         edge_weight_list.append(edge_weight_t)
#
#     return edge_index_list, edge_weight_list  # 现在返回的是 list，而不是 tensor
#
#
# class HiGP_GraphLearn(nn.Module):
#     """
#     整合 GraphLearn 结构学习的 HiGPTTSModel
#     """
#     def __init__(self, input_size, horizon, n_nodes, hidden_size, emb_size, levels,
#                  n_clusters, single_sample, mode='gated', skip_connection=False,
#                  top_down=False, output_size=None, rnn_size=None, ff_size=None,
#                  exog_size=0, temporal_layers=1, gnn_layers=1, temp_decay=0.99999,
#                  activation='elu', alpha=0.1, device='cpu'):
#         super(HiGP_GraphLearn, self).__init__()
#
#         # 初始化 HiGPTTSModel
#         self.higp = HiGPTTSModel(
#             input_size=input_size,
#             horizon=horizon,
#             n_nodes=n_nodes,
#             hidden_size=hidden_size,
#             emb_size=emb_size,
#             levels=levels,
#             n_clusters=n_clusters,
#             single_sample=single_sample,
#             skip_connection=skip_connection,
#             output_size=output_size,
#             exog_size=exog_size,
#             temporal_layers=temporal_layers,
#             activation=activation,
#             temp_decay=temp_decay
#         )
#
#         # 初始化 GraphLearn
#         self.graph_learn = GraphLearn(alpha=alpha, device=device)
#
#     def forward(self, x):
#         """
#         x: (B, T, V, F) - 神经元数据
#         return: 预测结果, 学习到的图结构, 损失
#         """
#         # ✅ 计算 S
#         S = self.graph_learn(x)
#
#         # ✅ 计算 edge_index 和 edge_weight
#         edge_index_list, edge_weight_list = compute_edge_index(S)
#
#         # ✅ 确保 `edge_index` 在 GPU / CPU 上
#         for b in range(len(edge_index_list)):
#             for t in range(len(edge_index_list[b])):
#                 edge_index_list[b][t] = edge_index_list[b][t].to(x.device)
#                 edge_weight_list[b][t] = edge_weight_list[b][t].to(x.device)
#
#         # ✅ 传入 HiGPTTSModel，按时间步处理
#         out_list = []
#         for t in range(S.shape[1]):  # 遍历时间步
#             out, aggregation_matrix, sizes, reg_losses = self.higp(
#                 x=x[:, t],  # 送入单个时间步的数据
#                 edge_index=edge_index_list[0][t],  # 假设 batch 内所有样本结构相同
#                 edge_weight=edge_weight_list[0][t]
#             )
#             out_list.append(out)
#
#         # 组合时间步结果
#         out = torch.stack(out_list, dim=1)
#
#         print("finish 1 iteration")
#
#         return out, S, self.graph_learn.diff_loss, self.graph_learn.f_norm_loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphLearn(nn.Module):
    def __init__(self, alpha=0.1, F=None, device='cpu'):
        super(GraphLearn, self).__init__()
        self.alpha = alpha
        self.device = device
        self.built = False

        if F is not None:
            self.register_parameter("a", nn.Parameter(torch.randn(F, 1).to(self.device)))
            self.built = True
        else:
            self.register_parameter("a", None)

    def build(self, F):
        """ 确保 self.a 被正确注册 """
        if not self.built:
            self.register_parameter("a", nn.Parameter(torch.randn(F, 1).to(self.device)))
            self.built = True

    def forward(self, x):
        B, T, V, F = x.shape
        if not self.built:
            self.build(F)

        S_list = []
        diff_loss_batch = 0.0

        for t in range(T):
            xt = x[:, t, :, :]
            x1 = xt.unsqueeze(2)
            x2 = xt.unsqueeze(1)
            diff = x1 - x2

            w_4d = torch.einsum('b i j f, f c-> b i j c', torch.abs(diff), self.a)
            w_3d = w_4d.squeeze(-1)
            tmpS = torch.exp(w_3d)
            S_ = F.normalize(tmpS, p=1, dim=-1)
            S_list.append(S_)

            diff_sq = diff.pow(2).sum(dim=-1)
            local_loss = (diff_sq * S_).sum(dim=[1, 2])
            diff_loss_batch += local_loss.mean()

        S = torch.stack(S_list, dim=1)

        self.diff_loss = diff_loss_batch
        self.f_norm_loss = self.alpha * (S.pow(2).sum())

        return S


def compute_edge_index(S):
    B, T, V, V = S.shape
    edge_index_tensor = []
    edge_weight_tensor = []

    for b in range(B):
        edge_index_t = []
        edge_weight_t = []

        for t in range(T):
            adj_matrix = S[b, t]
            edge_indices = torch.nonzero(adj_matrix, as_tuple=False).T

            if edge_indices.numel() == 0:
                edge_indices = torch.empty((2, 0), dtype=torch.long, device=S.device)
                edge_weights = torch.empty((0,), dtype=torch.float32, device=S.device)
            else:
                edge_weights = adj_matrix[edge_indices[0], edge_indices[1]]

            edge_index_t.append(edge_indices)
            edge_weight_t.append(edge_weights)

        edge_index_tensor.append(torch.stack(edge_index_t, dim=0))
        edge_weight_tensor.append(torch.stack(edge_weight_t, dim=0))

    return torch.stack(edge_index_tensor, dim=0), torch.stack(edge_weight_tensor, dim=0)


class HiGP_GraphLearn(nn.Module):
    def __init__(self, input_size, horizon, n_nodes, hidden_size, emb_size, levels,
                 n_clusters, single_sample, mode='gated', skip_connection=False,
                 top_down=False, output_size=None, rnn_size=None, ff_size=None,
                 exog_size=0, temporal_layers=1, gnn_layers=1, temp_decay=0.99999,
                 activation='elu', alpha=0.1, device='cpu'):
        super(HiGP_GraphLearn, self).__init__()

        from lib.nn.hierarchical.models.higp_tts_model import HiGPTTSModel  # 避免循环导入

        self.higp = HiGPTTSModel(
            input_size=input_size,
            horizon=horizon,
            n_nodes=n_nodes,
            hidden_size=hidden_size,
            emb_size=emb_size,
            levels=levels,
            n_clusters=n_clusters,
            single_sample=single_sample,
            skip_connection=skip_connection,
            output_size=output_size,
            exog_size=exog_size,
            temporal_layers=temporal_layers,
            activation=activation,
            temp_decay=temp_decay
        )

        self.graph_learn = GraphLearn(alpha=alpha, device=device)

    def forward(self, x):
        S = self.graph_learn(x)
        edge_index_list, edge_weight_list = compute_edge_index(S)

        edge_index_list = edge_index_list.to(x.device)
        edge_weight_list = edge_weight_list.to(x.device)

        out_list = []
        for t in range(S.shape[1]):
            out, _, _, _ = self.higp(x[:, t], edge_index_list[:, t], edge_weight_list[:, t])
            out_list.append(out)

        out = torch.stack(out_list, dim=1)

        return out, S, self.graph_learn.diff_loss, self.graph_learn.f_norm_loss
