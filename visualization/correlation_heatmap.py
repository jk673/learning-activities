"""Correlation heatmap between input features and target (residuary resistance)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from models.model_kriging import load_yacht_data


LABELS = [
    "Buoyancy pos.",
    "Prismatic coeff.",
    "Length-disp. ratio",
    "Beam-draught ratio",
    "Length-beam ratio",
    "Froude number",
    "Resid. resistance",
]


def plot_correlation_heatmap(save_path=None):
    X, y = load_yacht_data()
    data = np.column_stack([X, y])  # (308, 7)

    corr = np.corrcoef(data, rowvar=False)  # (7, 7)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(LABELS, fontsize=9)

    # 각 셀에 상관계수 표시
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            color = "white" if abs(corr[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson Correlation", fontsize=10)

    ax.set_title("Feature Correlation – Yacht Hydrodynamics", fontsize=13, pad=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    out = ROOT / "visualization" / "correlation_heatmap.png"
    plot_correlation_heatmap(save_path=out)
