"""Polynomial Response Surface model (Least Squares) for yacht hydrodynamics data."""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "yacht_hydrodynamics.data"


def load_yacht_data(path=DATA_PATH):
    """Load yacht hydrodynamics dataset."""
    raw = np.loadtxt(path)
    return raw[:, :6], raw[:, 6]


def build_polynomial_rs(X_train, y_train, degree=2):
    """Build and fit a polynomial response surface model.

    Expands features to polynomial terms (interactions + powers)
    then fits ordinary least squares.

    Returns:
        model: fitted LinearRegression
        poly: fitted PolynomialFeatures transformer
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_poly, y_train)
    return model, poly


def predict(model, poly, X, scaler):
    """Predict using the polynomial RS model."""
    X_scaled = scaler.transform(X)
    X_poly = poly.transform(X_scaled)
    return model.predict(X_poly)


def evaluate(model, poly, X_test, y_test, scaler):
    """Evaluate model on test set. Returns predictions, RMSE, and R2."""
    y_pred = predict(model, poly, X_test, scaler)

    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return y_pred, rmse, r2


if __name__ == "__main__":
    X, y = load_yacht_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    for degree in [2, 3]:
        print(f"\n{'='*50}")
        print(f"  Polynomial degree: {degree}")
        print(f"{'='*50}")

        model, poly = build_polynomial_rs(X_train_scaled, y_train, degree=degree)
        n_features = poly.n_output_features_
        print(f"  Expanded features: {X.shape[1]} → {n_features} terms")

        y_pred, rmse, r2 = evaluate(model, poly, X_test, y_test, scaler)

        mae = np.mean(np.abs(y_test - y_pred))
        max_err = np.max(np.abs(y_test - y_pred))
        mask = y_test > 0.05
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

        print(f"  R²:         {r2:.4f}")
        print(f"  RMSE:       {rmse:.4f}")
        print(f"  MAE:        {mae:.4f}")
        print(f"  MAPE:       {mape:.2f}%")
        print(f"  Max Error:  {max_err:.4f}")

        print(f"\n  {'Actual':>10s}  {'Predicted':>10s}  {'Error':>10s}")
        sort_idx = np.argsort(y_test)[:10]
        for i in sort_idx:
            err = y_pred[i] - y_test[i]
            print(f"  {y_test[i]:10.4f}  {y_pred[i]:10.4f}  {err:+10.4f}")
