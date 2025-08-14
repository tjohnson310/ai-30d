import os, json, math
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)


def make_data(n=1500, d=2, noise=1.0, seed=42):
    rng = np.random.RandomState(seed)
    w_true = np.array([3.5, -2.0][:d]); b_true = 7.0
    X = rng.randn(n, d); y = X @ w_true + b_true + rng.randn(n)*noise
    return X.astype(np.float32), y.astype(np.float32)


X, y = make_data()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr).unsqueeze(1)
Xte_t, yte_t = torch.from_numpy(Xte), torch.from_numpy(yte).unsqueeze(1)

model = nn.Sequential(nn.Linear(X.shape[1], 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.SGD(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

for epoch in range(600):
    opt.zero_grad()
    pred = model(Xtr_t)
    loss = loss_fn(pred, ytr_t)
    loss.backward()
    opt.step()

with torch.no_grad():
    pred_tr = model(Xtr_t).squeeze().numpy()
    pred_te = model(Xte_t).squeeze().numpy()

rmse_tr = math.sqrt(mean_squared_error(ytr, pred_tr))
rmse_te = math.sqrt(mean_squared_error(yte, pred_te))
print({"nn_rmse_train": rmse_tr, "nn_rmse_test": rmse_te})
