"""Kriging (Gaussian Process Regression) model for yacht hydrodynamics data."""

import numpy as np
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "yacht_hydrodynamics.data"


def load_yacht_data(path=DATA_PATH):
    """Load yacht hydrodynamics dataset.

    Returns:
        X: (n_samples, 6) input features
        y: (n_samples,) residuary resistance
    """
    raw = np.loadtxt(path)
    X = raw[:, :6]
    y = raw[:, 6]
    return X, y


def build_kriging(X_train, y_train):
    """Build and fit a Kriging model.

    Uses Matern 5/2 kernel with automatic hyperparameter optimization.
    """
    kernel = ConstantKernel() * Matern(nu=2.5, length_scale=np.ones(X_train.shape[1]))
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=42,
    )
    gpr.fit(X_train, y_train)
    return gpr


def evaluate(model, X_test, y_test, scaler):
    """Evaluate model on test set. Returns predictions, std, RMSE, and R2."""
    X_test_scaled = scaler.transform(X_test)
    y_pred, y_std = model.predict(X_test_scaled, return_std=True)

    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return y_pred, y_std, rmse, r2


if __name__ == "__main__":
    X, y = load_yacht_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("Fitting Kriging model...")
    model = build_kriging(X_train_scaled, y_train)

    print(f"Optimized kernel: {model.kernel_}")

    y_pred, y_std, rmse, r2 = evaluate(model, X_test, y_test, scaler)
    print(f"\nTest results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")

    print(f"\nSample predictions (first 10):")
    print(f"  {'Actual':>10s}  {'Predicted':>10s}  {'Std':>8s}")
    for actual, pred, std in zip(y_test[:10], y_pred[:10], y_std[:10]):
        print(f"  {actual:10.4f}  {pred:10.4f}  {std:8.4f}")
