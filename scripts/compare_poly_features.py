"""Compare polynomial RS: all 6 variables vs selected 4 variables."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.model_kriging import load_yacht_data
from models.model_polynomial_rs import build_polynomial_rs


# 선택 변수: buoyancy pos (0), prismatic coeff (1), beam-draught ratio (3), froude (5)
SELECTED_COLS = [0, 1, 3, 5]
SELECTED_NAMES = ["Buoyancy pos.", "Prismatic coeff.", "Beam-draught ratio", "Froude number"]
ALL_NAMES = [
    "Buoyancy pos.", "Prismatic coeff.", "Length-disp. ratio",
    "Beam-draught ratio", "Length-beam ratio", "Froude number",
]


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    max_err = np.max(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    mask = y_true > 0.05
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"R²": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape, "Max Error": max_err}


def run_experiment(X, y, label, feature_names, degree):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model, poly = build_polynomial_rs(X_train_scaled, y_train, degree=degree)
    X_test_scaled = scaler.transform(X_test)
    X_test_poly = poly.transform(X_test_scaled)
    y_pred = model.predict(X_test_poly)

    metrics = compute_metrics(y_test, y_pred)
    n_terms = poly.n_output_features_

    return metrics, n_terms, y_test, y_pred


if __name__ == "__main__":
    X_all, y = load_yacht_data()
    X_sel = X_all[:, SELECTED_COLS]

    for degree in [2, 3]:
        print(f"\n{'='*70}")
        print(f"  POLYNOMIAL DEGREE {degree}")
        print(f"{'='*70}")

        m_all, t_all, yt_all, yp_all = run_experiment(
            X_all, y, "All 6 vars", ALL_NAMES, degree
        )
        m_sel, t_sel, yt_sel, yp_sel = run_experiment(
            X_sel, y, "Selected 4 vars", SELECTED_NAMES, degree
        )

        print(f"\n  {'Metric':<12s}  {'All 6 vars':>12s}  {'Selected 4':>12s}  {'Winner':>10s}")
        print(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*10}")

        for key in ["R²", "RMSE", "MAE", "MAPE", "Max Error"]:
            v_all, v_sel = m_all[key], m_sel[key]
            if key == "R²":
                winner = "All 6" if v_all > v_sel else "Sel 4"
            else:
                winner = "All 6" if v_all < v_sel else "Sel 4"
            unit = "%" if key == "MAPE" else ""
            print(f"  {key:<12s}  {v_all:>11.4f}{unit}  {v_sel:>11.4f}{unit}  {winner:>10s}")

        print(f"  {'Poly terms':<12s}  {t_all:>12d}  {t_sel:>12d}")
