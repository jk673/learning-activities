import numpy as np


def branin(x1, x2, a=1.0, b=5.1 / (4 * np.pi**2), c=5.0 / np.pi,
           r=6.0, s=10.0, t=1.0 / (8 * np.pi)):
    """Branin-Hoo benchmark function.

    Standard test function for global optimization.
    Has three global minima with f(x*) = 0.397887.

    Global minima at:
        (x1, x2) = (-pi, 12.275)
        (x1, x2) = (pi, 2.275)
        (x1, x2) = (9.42478, 2.475)

    Standard domain: x1 in [-5, 10], x2 in [0, 15].

    Args:
        x1: First input variable.
        x2: Second input variable.
        a, b, c, r, s, t: Function parameters (defaults are standard values).

    Returns:
        Scalar function value.
    """
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s


def branin_grid(x1_range=(-5, 10), x2_range=(0, 15), n_points=100):
    """Generate a grid of Branin function values.

    Args:
        x1_range: Tuple of (min, max) for x1.
        x2_range: Tuple of (min, max) for x2.
        n_points: Number of points per axis.

    Returns:
        Tuple of (X1, X2, Y) meshgrid arrays.
    """
    x1 = np.linspace(x1_range[0], x1_range[1], n_points)
    x2 = np.linspace(x2_range[0], x2_range[1], n_points)
    X1, X2 = np.meshgrid(x1, x2)
    Y = branin(X1, X2)
    return X1, X2, Y


def branin_random_samples(n_samples=100, x1_range=(-5, 10), x2_range=(0, 15),
                          seed=None):
    """Generate random samples of the Branin function.

    Args:
        n_samples: Number of random samples.
        x1_range: Tuple of (min, max) for x1.
        x2_range: Tuple of (min, max) for x2.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, Y) where X is (n_samples, 2) and Y is (n_samples,).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(x1_range[0], x1_range[1], n_samples)
    x2 = rng.uniform(x2_range[0], x2_range[1], n_samples)
    X = np.column_stack([x1, x2])
    Y = branin(x1, x2)
    return X, Y


if __name__ == "__main__":
    # Quick demo
    X1, X2, Y = branin_grid(n_points=50)
    print(f"Grid shape: {Y.shape}")
    print(f"Min value: {Y.min():.6f}")

    X, Y = branin_random_samples(n_samples=10, seed=42)
    print(f"\nRandom samples:")
    for i in range(len(Y)):
        print(f"  x=({X[i, 0]:7.3f}, {X[i, 1]:7.3f})  f={Y[i]:.4f}")
