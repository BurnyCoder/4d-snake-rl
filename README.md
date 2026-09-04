# 4D Snake + reinforcement learning

An N-dimensional snake game (default 4D: a `4 x 4 x 4 x 4` board with 256 cells and 8 move
directions) and a MaskablePPO agent trained to *complete* it, i.e. grow until the snake fills
every cell.

Work in progress: the game core, baselines, training and reports land in successive pull requests.
Setup and conventions live in [AGENTS.md](AGENTS.md); the full methodology, usage and results are
written here as they are produced.

## Quick start

```bash
uv sync --all-groups
uv run pytest -m "not slow and not gpu"
```
