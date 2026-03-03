"""Pearson vs Spearman vs Mutual Information comparison for yacht hydrodynamics."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from models.model_kriging import load_yacht_data


FEATURE_LABELS = [
    "Buoyancy pos.",
    "Prismatic coeff.",
    "Length-disp. ratio",
    "Beam-draught ratio",
    "Length-beam ratio",
    "Froude number",
]


def compute_metrics(X, y):
    n_features = X.shape[1]

    # Pearson
    pearson = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(n_features)])

    # Spearman
    spearman = np.array([spearmanr(X[:, i], y).correlation for i in range(n_features)])

    # Mutual Information (평균 3회로 안정화)
    mi_runs = np.array([
        mutual_info_regression(X, y, random_state=seed)
        for seed in range(3)
    ])
    mi = mi_runs.mean(axis=0)
    # 0~1 스케일로 정규화 (비교 용이)
    mi_norm = mi / mi.max()

    return pearson, spearman, mi_norm, mi


def plot_comparison(save_path=None):
    X, y = load_yacht_data()
    pearson, spearman, mi_norm, mi_raw = compute_metrics(X, y)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    x_pos = np.arange(len(FEATURE_LABELS))
    bar_kw = dict(edgecolor="black", linewidth=0.5)

    # --- Pearson ---
    ax = axes[0]
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in pearson]
    bars = ax.barh(x_pos, pearson, color=colors, **bar_kw)
    ax.set_xlim(-1, 1)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(FEATURE_LABELS)
    ax.set_xlabel("Pearson r (linear)")
    ax.set_title("Pearson Correlation")
    ax.axvline(0, color="gray", linewidth=0.8)
    for i, v in enumerate(pearson):
        ax.text(v + 0.03 * np.sign(v), i, f"{v:.2f}", va="center", fontsize=9)

    # --- Spearman ---
    ax = axes[1]
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in spearman]
    ax.barh(x_pos, spearman, color=colors, **bar_kw)
    ax.set_xlim(-1, 1)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(FEATURE_LABELS)
    ax.set_xlabel("Spearman ρ (monotonic)")
    ax.set_title("Spearman Rank Correlation")
    ax.axvline(0, color="gray", linewidth=0.8)
    for i, v in enumerate(spearman):
        ax.text(v + 0.03 * np.sign(v), i, f"{v:.2f}", va="center", fontsize=9)

    # --- Mutual Information ---
    ax = axes[2]
    ax.barh(x_pos, mi_norm, color="#8e44ad", **bar_kw)
    ax.set_xlim(0, 1.15)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(FEATURE_LABELS)
    ax.set_xlabel("Normalized MI (any dependency)")
    ax.set_title("Mutual Information")
    for i, (v, raw) in enumerate(zip(mi_norm, mi_raw)):
        ax.text(v + 0.02, i, f"{v:.2f} ({raw:.2f})", va="center", fontsize=9)

    fig.suptitle("Feature Relevance to Residuary Resistance", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    out = ROOT / "visualization" / "nonlinear_correlation.png"
    plot_comparison(save_path=out)
