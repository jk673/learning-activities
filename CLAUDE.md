# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a playground for training and experimenting with machine learning models from various data sources. Treat it like a whiteboard — explore ideas freely, prototype quickly, and iterate. The codebase grows organically as new learning activities, datasets, and models are added.

## Commands

```bash
# Install dependencies
uv sync

# Run a Python script
uv run python functions/branin.py
uv run python visualization/contour_2d.py

# Add a new dependency
uv add <package>
```

No test framework, linter, or formatter is currently configured.

## Architecture

```
data/            → Raw datasets (CSV-like, space-separated, etc.)
functions/       → Reusable mathematical/ML functions (e.g., benchmark optimization functions)
visualization/   → Plotting scripts and generated figures
models/          → Trained model artifacts (empty, to be populated)
scripts/         → Utility/runner scripts (empty, to be populated)
docs/            → Documentation (empty, to be populated)
logs/            → Runtime logs
```

**Pattern:** `functions/` provides computation, `visualization/` consumes it for plotting, `data/` holds raw inputs, `models/` will hold outputs. Scripts in `scripts/` tie things together.

## Current Contents

- **Branin-Hoo function** (`functions/branin.py`): Benchmark optimization test function with grid and random sampling helpers. Used by `visualization/contour_2d.py` for contour plots.
- **Yacht hydrodynamics dataset** (`data/yacht_hydrodynamics.data`): 6 input features → 1 output (residuary resistance). Described in `data/yatch_hydrodynamics.cards`.

## Conventions

- **Python ≥ 3.10**, managed with `uv`
- Only `numpy` is a declared dependency; `matplotlib` is used in visualization but not yet in `pyproject.toml` — add it when needed
- Scripts include `if __name__ == "__main__":` demo blocks for quick testing
- No strict code style enforced — keep it readable and exploratory
