# Documentation

| Document | Purpose |
|---|---|
| [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md) | Motivation, design, reproducibility workflow, and presentation guidance |
| [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md) | Verified results, ablations, acceptance criteria, and claim boundaries |
| [`Research review`](../output/pdf/NetAnomaly_OW_Research_Review.pdf) | Rendered architecture, metrics, and operational analysis |

## Notebooks

| Notebook | Purpose |
|---|---|
| [`01_data_exploration.ipynb`](../notebooks/01_data_exploration.ipynb) | Application balance, traffic distributions, temporal activity, and packet profiles |
| [`02_model_evaluation.ipynb`](../notebooks/02_model_evaluation.ipynb) | Model metrics, novelty trade-offs, flag composition, and early-flow performance |
| [`03_model_training.ipynb`](../notebooks/03_model_training.ipynb) | Reproducible neural training commands, loss curves, and convergence summary |

Generated per-flow predictions and model-specific JSON reports are written under `reports/` during
local experiments and are intentionally excluded from Git.
