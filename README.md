# learning-activities

## Quick `uv` setup

This project uses `uv` for Python environment/dependency management.

1. Install dependencies:
   - `uv sync`
2. Run scripts with:
   - `uv run python functions/branin.py`

## Notes

`functions/branin.py` currently depends on:
- `numpy`

If you prefer a dedicated environment path:
1. `uv venv`
2. `source .venv/bin/activate` (or use `\.venv\Scripts\activate` on Windows)
3. `uv sync`
