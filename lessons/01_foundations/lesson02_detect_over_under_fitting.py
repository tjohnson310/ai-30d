import json, os, math, random
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')
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


def train_linear_gd_safe(X, y, lr=1e-3, epochs=3000, l2=0.0, clip_norm=1.0, tol=1e-9):
    """
    Numerically robust GD:
    - smaller lr (defaul 1e-3
    - optional L2 (weight decay)
    - gradient clipping by L2 norm
    - early stop on tiny loss change
    - NaN/Inf guards
    :param X:
    :param y:
    :param lr:
    :param epochs:
    :param l2:
    :param clip_norm:
    :param tol:
    :return: w, b, losses
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    losses = []

    prev_loss = np.inf
    for t in range(epochs):
        yhat = X @ w + b
        err = yhat - y

        # MSE + L2 penalty
        loss = (err @ err) / n + l2 * (w @ w)
        if not np.isfinite(loss):
            break

        # gradients
        grad_w = (2.0/n) * (X.T @ err) + 2.0 * l2 * w
        grad_b = (2.0/n) * err.sum()

        # clip grads
        if clip_norm is not None and clip_norm > 0:
            gnorm = np.linalg.norm(grad_w)
            if gnorm > clip_norm:
                grad_w *= (clip_norm / max(gnorm, 1e-12))

        # update
        w -= lr * grad_w
        b -= lr * grad_b

        losses.append(loss)

        # early stop if barely improving
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss

    return w, b, losses


def expand_and_scale(X_tr_s, X_te_s, degree):
    """
    Expand with PolynomialFeatures, then StandardScale the expanded matrix.
    Returns transformed train/test plus the fitted transformers.
    :param X_tr_s:
    :param X_te_s:
    :param degree:
    :return: X_tr_ps, X_te_ps, poly, scaler
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_tr_p = poly.fit_transform(X_tr_s)
    X_te_p = poly.transform(X_te_s)

    scaler = StandardScaler()
    X_tr_ps = scaler.fit_transform(X_tr_p)
    X_te_ps = scaler.transform(X_te_p)
    return X_tr_ps, X_te_ps, poly, scaler


def fit_and_report(X_tr_s, X_te_s, y_tr, y_te, degree):
    # 1) Expand features with degree-d polynomials ( no bias term - let the model handle intercept)
    X_tr_ps, X_te_ps, _, _ = expand_and_scale(X_tr_s, X_te_s, degree)
    model = LinearRegression()
    model.fit(X_tr_ps, y_tr)
    y_tr_pred = model.predict(X_tr_ps)
    y_te_pred = model.predict(X_te_ps)

    rmse_tr = np.sqrt(mean_squared_error(y_tr, y_tr_pred))
    rmse_te = np.sqrt(mean_squared_error(y_te, y_te_pred))
    print(f"[OLS] Degree {degree} | Train RMSE: {rmse_tr:.3f} | Test RMSE: {rmse_te:.3f}")
    return model


def fit_gd_and_report(X_tr_s, X_te_s, y_tr, y_te,degree, lr=1e-3, epochs=3000, l2=1e-4, clip_norm=1.0):
    Xg_tr, Xg_te, _, _ = expand_and_scale(X_tr_s, X_te_s, degree)

    w_hat, b_hat, losses = train_linear_gd_safe(
        Xg_tr, y_tr,
        lr=lr,
        epochs=epochs,
        l2=l2,
        clip_norm=clip_norm,
        tol=1e-9
    )

    y_tr_pred = Xg_tr @ w_hat + b_hat
    y_te_pred = Xg_te @ w_hat + b_hat
    rmse_tr = np.sqrt(((y_tr_pred - y_tr)**2).mean())
    rmse_te = np.sqrt(((y_te_pred - y_te) ** 2).mean())
    print(f"[GD] Degree {degree} | Train RMSE: {rmse_tr:.3f} | Test RMSE: {rmse_te:.3f}")
    return w_hat, b_hat, losses

@dataclass
class RunResult:
    rmse_tr: float
    rmse_te: float


def eval_rmse(y_true_tr, y_pred_tr, y_true_te, y_pred_te) -> RunResult:
    rmse_tr = np.sqrt(mean_squared_error(y_true_tr, y_pred_tr))
    rmse_te = np.sqrt(mean_squared_error(y_true_te, y_pred_te))
    return RunResult(rmse_tr, rmse_te)


def plot_rmse_vs_degree(X_tr_s, X_te_s, y_tr, y_te):
    degrees = [1, 2, 3, 4, 5]
    ols_train_rmse, ols_test_rmse = [], []
    gd_train_rmse, gd_test_rmse = [], []

    for d in degrees:
        X_tr_ps, X_te_ps, *_ = expand_and_scale(X_tr_s, X_te_s, degree=d)
        ols = LinearRegression()
        ols.fit(X_tr_ps, y_tr)
        ols_train_rmse.append(np.sqrt(mean_squared_error(y_tr, ols.predict(X_tr_ps))))
        ols_test_rmse.append(np.sqrt(mean_squared_error(y_te, ols.predict(X_te_ps))))

        w_hat, b_hat, losses = train_linear_gd_safe(
            X_tr_ps, y_tr,
            lr=3e-3,
            epochs=800,
            l2=1e-4,
            clip_norm=1.0
        )
        y_tr_pred = X_tr_ps @ w_hat + b_hat
        y_te_pred = X_te_ps @ w_hat + b_hat
        gd_train_rmse.append(np.sqrt(mean_squared_error(y_tr, y_tr_pred)))
        gd_test_rmse.append(np.sqrt(mean_squared_error(y_te, y_te_pred)))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(degrees, ols_train_rmse, 'o-', label='OLS Train RMSE')
    plt.plot(degrees, ols_test_rmse, 'o-', label='OLS Test RMSE')
    plt.plot(degrees, gd_train_rmse, 's-', label='GD Train RMSE')
    plt.plot(degrees, gd_test_rmse, 's-', label='GD Test RMSE')

    # Annotate variance-like uptick at degree 5 for GD
    plt.annotate(
        "Variance-like uptick\n(test RMSE rises vs deg 4)",
        xy=(5, gd_test_rmse[-1]),
        xytext=(3.7, 1.65),
        arrowprops=dict(arrowstyle="->", lw=3, color='black'),
        fontsize=9,
        color='red'
    )

    # Annotate underfitting gap at degree 1
    plt.annotate(
        "Underfitting gap (GD >> OLS)",
        xy=(1, gd_train_rmse[0]),
        xytext=(1.5, 1.75),
        arrowprops=dict(arrowstyle="->", lw=3, color='black'),
        fontsize=9
    )

    plt.xlabel('Polynomial Degree')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Polynomial Degree (OLS vs GD)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save alongside your other artifacts
    try:
        plt.savefig(f"{ART_DIR}/rmse_vs_degree.png", dpi=150)
        print(f"Saved plot to {ART_DIR}/rmse_vs_degree.png")
    except Exception:
        pass

    plt.show()

    # Quick numeric dump for the console
    print("Degrees:", degrees)
    print("OLS   Train:", [round(x, 3) for x in ols_train_rmse])
    print("OLS   Test :", [round(x, 3) for x in ols_test_rmse])
    print("GD    Train:", [round(x, 3) for x in gd_train_rmse])
    print("GD    Test :", [round(x, 3) for x in gd_test_rmse])


def main():
    X, y, w_true, b_true = make_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    X_tr_s, X_te_s, _, _ = standardize(X_tr, X_te)

    plot_rmse_vs_degree(X_tr_s, X_te_s, y_tr, y_te)


if __name__ == "__main__":
    main()
