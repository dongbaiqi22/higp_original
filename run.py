import torch
import numpy as np

from STcal.FC_block import HiGP_GraphLearn

# 1. 读取数据
data = np.load("/Users/dongbaiqi/Desktop/higp-main/data/fish_data.npz")
X = torch.tensor(data["Fold_data"], dtype=torch.float32)  # (10, 24, 50, epoch_length)
Y = torch.tensor(data["Fold_label"], dtype=torch.float32)  # (10, 24, 3)

# 2. 变换形状 (10, 24, 50, epoch_length) -> (240, epoch_length, 50, 1)
B, epochs, V, T = X.shape
X_graphlearn = X.reshape(B * epochs, T, V, 1)  # (240, epoch_length, 50, 1)

# 3. 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HiGP_GraphLearn(
    input_size=T,  # epoch_length
    horizon=1,  # 预测 1 步
    n_nodes=V,  # 50 个神经元
    hidden_size=64,
    emb_size=16,
    levels=3,
    n_clusters=20,
    single_sample=False,
    skip_connection=True,
    gnn_layers=2,
    activation='elu',
    alpha=0.1,
    device=device
).to(device)

# 4. 训练
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    out, S, diff_loss, f_norm_loss = model(X_graphlearn.to(device))

    loss = loss_fn(out, Y.reshape(B * epochs, -1).to(device)) + diff_loss + f_norm_loss
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
