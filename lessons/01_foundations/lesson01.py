import json, os, math, random
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)


@dataclass
class Metrics:
    rmse_train: float
    rmse_test: float
    w_est: list
    b_est: float
    w_true: list
    b_true: float


def make_data(n_samples=1500, n_features=2, noise_std=1.0, seed=42):
    rng = np.random.RandomState(seed)
    # ground truth weights/bias
    w_true = np.array([3.5, -2.0][:n_features])
    b_true = 7.0
    X = rng.randn(n_samples, n_features)
    y = X @ w_true + b_true + rng.randn(n_samples) * noise_std
    return X, y, w_true, b_true


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


def main():
    X, y, w_true, b_true = make_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    # Standardize for stable GD
    X_tr_s, X_te_s, mu, std = standardize(X_tr, X_te)

    # ---- From-scratch GD ----
    w_hat, b_hat, losses = train_linear_gd(X_tr_s, y_tr, lr=0.05, epochs=600)
    # undo standardization to compare with true params
    w_hat_unscaled = w_hat / std
    b_hat_unscaled = b_hat - (mu / std) @ w_hat + 0.0

    rmse_tr = math.sqrt(mean_squared_error(y_tr, X_tr @ w_hat_unscaled + b_hat_unscaled))
    rmse_te = math.sqrt(mean_squared_error(y_te, X_te @ w_hat_unscaled + b_hat_unscaled))

    # ---- scikit-learn baseline ----
    linreg = LinearRegression()
    linreg.fit(X_tr, y_tr)
    rmse_tr_skl = math.sqrt(mean_squared_error(y_tr, linreg.predict(X_tr)))
    rmse_te_skl = math.sqrt(mean_squared_error(y_te, linreg.predict(X_te)))

    # ---- Plot loss curve ----
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Lesson 1: GD Loss Curve")
    loss_path = os.path.join(ART_DIR, "lesson01_loss.png")
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")

    # ---- Save metrics ----
    scratch_metrics = Metrics(
        rmse_train=rmse_tr,
        rmse_test=rmse_te,
        w_est=list(w_hat_unscaled),
        b_est=float(b_hat_unscaled),
        w_true=list(w_true),
        b_true=float(b_true)
    )
    result = {
        "scratch": asdict(scratch_metrics),
        "sklearn": {"rmse_train": rmse_tr_skl, "rmse_test": rmse_te_skl, "coef_": linreg.coef_.tolist(),
                    "intercept_": float(linreg.intercept_)}
    }
    with open(os.path.join(ART_DIR, "lesson01_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n==== RESULTS ====")
    print(json.dumps(result, indent=2))
    print(f"\nSaved loss curve to: {loss_path}")
    print("Done.")


if __name__ == "__main__":
    main()


















