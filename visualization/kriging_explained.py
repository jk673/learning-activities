"""Kriging 작동 원리 시각화.

1. 1D slice: Froude 수 변화에 따른 예측 곡선 + 불확실성 밴드
2. Ground truth vs Prediction 산점도
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.model_kriging import load_yacht_data, build_kriging


def main(save_path=None):
    X, y = load_yacht_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = build_kriging(X_train_scaled, y_train)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # =========================================================
    # Panel 1: 1D slice — 하나의 선체 형상, Froude만 변화
    # =========================================================
    ax = axes[0]

    # 첫 번째 선체 형상 선택 (데이터 row 0의 형상 변수)
    hull = X[0, :5]  # [-2.3, 0.568, 4.78, 3.99, 3.17]

    # 이 선체 형상에 해당하는 실제 데이터 추출
    hull_mask = np.all(np.isclose(X[:, :5], hull, atol=0.01), axis=1)
    X_hull = X[hull_mask]
    y_hull = y[hull_mask]
    froude_actual = X_hull[:, 5]

    # Froude 연속 범위에서 Kriging 예측
    froude_dense = np.linspace(0.12, 0.46, 200)
    X_query = np.column_stack([
        np.tile(hull, (len(froude_dense), 1)),
        froude_dense,
    ])
    X_query_scaled = scaler.transform(X_query)
    y_mean, y_std = model.predict(X_query_scaled, return_std=True)

    # 예측 곡선 + 불확실성 밴드
    ax.fill_between(froude_dense, y_mean - 2 * y_std, y_mean + 2 * y_std,
                    alpha=0.2, color="#3498db", label="95% confidence")
    ax.plot(froude_dense, y_mean, color="#2c3e50", linewidth=2, label="Kriging mean")

    # 학습/테스트 데이터 분리해서 표시
    train_mask = np.isin(np.where(hull_mask)[0],
                         train_test_split(range(len(y)), test_size=0.2, random_state=42)[0])
    ax.scatter(froude_actual[train_mask], y_hull[train_mask],
               color="#e74c3c", s=60, zorder=5, edgecolors="black",
               linewidths=0.8, label="Train data")
    ax.scatter(froude_actual[~train_mask], y_hull[~train_mask],
               color="#2ecc71", s=80, zorder=5, edgecolors="black",
               linewidths=0.8, marker="D", label="Test data")

    ax.set_xlabel("Froude Number", fontsize=10)
    ax.set_ylabel("Residuary Resistance", fontsize=10)
    ax.set_title("Kriging 1D Slice\n(single hull geometry, varying Froude)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # =========================================================
    # Panel 2: 같은 Froude, 다른 선체 → 불확실성 차이
    # =========================================================
    ax = axes[1]

    # Froude=0.3 고정, 각 선체 형상별 예측
    froude_fixed = 0.3
    unique_hulls = np.unique(X[:, :5], axis=0)

    actuals, preds, stds = [], [], []
    for h in unique_hulls:
        mask = np.all(np.isclose(X[:, :5], h, atol=0.01), axis=1) & \
               np.isclose(X[:, 5], froude_fixed, atol=0.01)
        if mask.any():
            xq = np.concatenate([h, [froude_fixed]]).reshape(1, -1)
            xq_s = scaler.transform(xq)
            pred, std = model.predict(xq_s, return_std=True)
            actuals.append(y[mask][0])
            preds.append(pred[0])
            stds.append(std[0])

    actuals, preds, stds = np.array(actuals), np.array(preds), np.array(stds)
    sort_idx = np.argsort(actuals)

    x_pos = np.arange(len(actuals))
    ax.bar(x_pos, actuals[sort_idx], width=0.4, align="edge", color="#bdc3c7",
           edgecolor="black", linewidth=0.5, label="Actual")
    ax.bar(x_pos + 0.4, preds[sort_idx], width=0.4, align="edge", color="#3498db",
           edgecolor="black", linewidth=0.5, label="Kriging pred.")
    ax.errorbar(x_pos + 0.6, preds[sort_idx], yerr=2 * stds[sort_idx],
                fmt="none", color="#e74c3c", capsize=3, linewidth=1.2, label="±2σ")

    ax.set_xlabel("Hull configuration (sorted by actual)", fontsize=10)
    ax.set_ylabel("Residuary Resistance", fontsize=10)
    ax.set_title(f"Per-Hull Prediction @ Froude={froude_fixed}\n(different hulls, same speed)",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # =========================================================
    # Panel 3: Prediction vs Ground Truth (전체 테스트셋)
    # =========================================================
    ax = axes[2]

    X_test_scaled = scaler.transform(X_test)
    y_pred, y_std = model.predict(X_test_scaled, return_std=True)

    ax.errorbar(y_test, y_pred, yerr=2 * y_std, fmt="o", markersize=4,
                color="#3498db", ecolor="#e74c3c", elinewidth=0.8,
                capsize=2, alpha=0.7, label="Predictions ±2σ")

    lims = [min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Perfect fit")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    ax.text(0.05, 0.92, f"R² = {r2:.4f}\nRMSE = {rmse:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Actual Resistance", fontsize=10)
    ax.set_ylabel("Predicted Resistance", fontsize=10)
    ax.set_title("Ground Truth vs Prediction\n(test set, 20%)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    out = ROOT / "visualization" / "kriging_explained.png"
    main(save_path=out)
