import numpy as np
import scipy.io as scio
from os import path

from attr.validators import min_len
from scipy import signal

import torch

data = np.load("/Users/dongbaiqi/Desktop/higp-main/data/ISRUC_S3/Feature_1.npz")

path_output    = './data/ISRUC_S3/'

print(data["train_feature"].shape)

subject_one = data["train_feature"][0:764, :, :]

print("subject 1 shape: ",subject_one.shape)

subject_one_reshaped = subject_one.transpose(1, 0, 2).reshape(10, -1, 1)

print("Reshaped subject shape:", subject_one_reshaped.shape)

np.savez(path.join(path_output, 'subject_1_data.npz'),
    subject = subject_one_reshaped
)

print("subject 1 data has been saved successfully!")


import numpy as np
import torch
import torch.nn.functional as F
import os

os.makedirs(path_output, exist_ok=True)

loaded_data = np.load(path.join(path_output, 'subject_1_data.npz'))
subject_one_reshaped = loaded_data["subject"]  # (10, 195584, 1)

subject_one_reshaped = subject_one_reshaped.squeeze(-1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = torch.tensor(subject_one_reshaped, dtype=torch.float32, device=device)

X_norm = F.normalize(X, p=2, dim=1)
connectivity_matrix = torch.mm(X_norm, X_norm.T)

connectivity_matrix_np = connectivity_matrix.cpu().numpy()

np.savez(path.join(path_output, 'connectivity_matrix.npz'), connectivity=connectivity_matrix_np)

print("Connectivity matrix has been saved successfully!")


# def adjacency_to_edge_list(adj_matrix):
#
#     rows, cols = np.nonzero(adj_matrix)
#     weights = adj_matrix[rows, cols]
#
#     # 构建两个矩阵
#     edge_list = np.vstack((rows, cols))  # 2 x num_edges
#     weight_list = weights.reshape(1, -1)  # 1 x num_edges
#
#     return edge_list, weight_list
#
#
# adjacency_to_edge_list(connectivity_matrix_np)






