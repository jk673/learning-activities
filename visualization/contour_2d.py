import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from functions.branin import branin, branin_grid


def plot_branin_contour(n_points=200, levels=30, save_path=None):
    """Plot a 2D filled contour of the Branin function."""
    X1, X2, Y = branin_grid(n_points=n_points)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(X1, X2, Y, levels=levels, cmap="viridis")
    cs = ax.contour(X1, X2, Y, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
    fig.colorbar(cf, ax=ax, label="f(x1, x2)")

    # Mark the three global minima
    minima = [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]
    for x1, x2 in minima:
        ax.plot(x1, x2, "r*", markersize=12)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Branin Function – 2D Contour")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_branin_contour(save_path="visualization/branin_contour.png")
