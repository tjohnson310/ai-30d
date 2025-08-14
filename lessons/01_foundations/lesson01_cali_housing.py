import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing


def standardize(X_train, X_test):
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mu)/std, (X_test - mu)/std, mu, std


def train_linear_gd(X, y, lr=0.05, epochs=500):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = []
    for t in range(epochs):
        yhat = X @ w + b
        err = yhat - y
        loss = (err**2).mean()
        # gradients
        grad_w = (2.0/n) * (X.T @ err)
        grad_b = (2.0/n) * err.sum()
        # update
        w -= lr * grad_w
        b -= lr * grad_b
        losses.append(loss)
    return w, b, losses


def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def main():
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize for stable GD
    X_tr_s, X_te_s, mu, std = standardize(X_tr, X_te)

    # ---- From-scratch GD ----
    w_hat, b_hat, losses = train_linear_gd(X_tr_s, y_tr, lr=0.05, epochs=600)
    # undo standardization to compare with true params
    w_hat_unscaled = w_hat / std
    b_hat_unscaled = b_hat - (mu / std) @ w_hat + 0.0

    # ---- Predictions + metrics (our GD model, evaluated in original feature space) ----
    y_pred_tr_gd = X_tr @ w_hat_unscaled + b_hat_unscaled
    y_pred_te_gd = X_te @ w_hat_unscaled + b_hat_unscaled
    rmse_tr_gd = rmse(y_tr, y_pred_tr_gd)
    rmse_te_gd = rmse(y_te, y_pred_te_gd)

    # ---- scikit-learn baseline ----
    ols = LinearRegression()
    ols.fit(X_tr, y_tr)
    w_sklearn = ols.coef_
    b_sklearn = ols.intercept_

    y_pred_tr_sk = ols.predict(X_tr)
    y_pred_te_sk = ols.predict(X_te)
    rmse_tr_sk = rmse(y_tr, y_pred_tr_sk)
    rmse_te_sk = rmse(y_te, y_pred_te_sk)

    # ---- Report ----
    print("\n==== California Housing: GD vs sklearn LinearRegression ====")
    print(f"Train RMSE | GD: {rmse_tr_gd:.4f} | sklearn: {rmse_tr_sk:.4f}")
    print(f"Test RMSE | GD: {rmse_te_gd:.4f} | sklearn: {rmse_te_sk:.4f}")

    print("\nFeature             |   GD_coef  |  SK_coef")
    print("--------------------+------------+----------")
    for name, c_gd, c_sk in zip(feature_names, w_hat_unscaled, w_sklearn):
        print(f"{name:19s} | {c_gd:10.6f} | {c_sk:9.6f}")
    print(f"\nBias (intercept)    | {b_hat_unscaled:10.6f} | {b_sklearn:9.6f}")

    # (Optional) quick sanity check on loss decreasing
    if len(losses) >= 2:
        print(f"\nLoss start: {losses[0]:.6f} --> Loss end: {losses[-1]:.6f}")

    '''
    ---- Expected Results ---- 
    ==== California Housing: GD vs sklearn LinearRegression ====
    Train RMSE | GD: 0.7197 | sklearn: 0.7197
    Test RMSE | GD: 0.7458 | sklearn: 0.7456
    
    Feature             |   GD_coef  |  SK_coef
    --------------------+------------+----------
    MedInc              |   0.451701 |  0.448675
    HouseAge            |   0.009894 |  0.009724
    AveRooms            |  -0.126928 | -0.123323
    AveBedrms           |   0.797241 |  0.783145
    Population          |  -0.000001 | -0.000002
    AveOccup            |  -0.003556 | -0.003526
    Latitude            |  -0.410010 | -0.419792
    Longitude           |  -0.423579 | -0.433708
    
    Bias (intercept)    | -36.173999 | -37.023278
    
    Loss start: 5.629742 --> Loss end: 0.517983
    '''


if __name__ == "__main__":
    main()

