"""Froude number vs Residuary resistance scatter plot."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from models.model_kriging import load_yacht_data


def plot_froude_vs_resistance(save_path=None):
    X, y = load_yacht_data()
    froude = X[:, 5]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(froude, y, s=15, alpha=0.6, edgecolors="black", linewidths=0.3)

    ax.set_xlabel("Froude Number", fontsize=11)
    ax.set_ylabel("Residuary Resistance", fontsize=11)
    ax.set_title("Froude Number vs Residuary Resistance", fontsize=13)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    out = ROOT / "visualization" / "froude_vs_resistance.png"
    plot_froude_vs_resistance(save_path=out)
