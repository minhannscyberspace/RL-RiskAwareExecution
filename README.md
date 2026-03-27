# RL-RiskAwareExecution

I built RL-RiskAwareExecution with the goal for it to be a quant research platform for single-order optimal execution.  
It is designed to test whether reinforcement learning execution policies can improve transaction quality relative to rule-based baselines under explicit market frictions and operational constraints.

The system is intentionally structured as an auditable pipeline, with strict controls for data leakage, reproducibility, and out-of-sample evaluation.


## Important Disclaimer

- This repository is for research and educational purposes only.
- Nothing in this project constitutes investment advice, trading advice, or a recommendation to transact in any instrument.
- This codebase is not connected to live markets and is not configured for live trading.
- Any deployment to production trading infrastructure requires separate controls, model governance, compliance review, and operational risk sign-off.


## Overview
Given a parent order (buy-side framing), the objective is to minimize implementation shortfall while respecting practical execution constraints:

- limited participation per interval,
- finite execution horizon,
- market impact and fees/slippage,
- completion requirements at horizon end.

The platform evaluates learned policies against established scheduling baselines:

- **TWAP**
- **VWAP**
- **POV**

## Scope and Non-Goals

### In scope
- Research-grade simulation for execution policy learning.
- Walk-forward and regime-aware benchmark comparisons.
- Deterministic artifact generation (config snapshots, metrics, reports).
- CPU-friendly training/evaluation workflows suitable for iterative research.

### Not in scope 
- Live execution or broker connectivity.
- Full limit order book simulation and queue dynamics.
- Multi-asset cross-impact modeling.
- Compliance workflows for production deployment.

##  Contributions

- Formulates execution as a constrained RL problem with explicit cost and completion penalties
- Introduces a risk-aware reward structure aligned with implementation shortfall
- Provides a reproducible evaluation pipeline with walk-forward validation
- Demonstrates how RL adapts execution intensity to market conditions vs static schedules

## Core Research Architecture

### Pipeline
`data -> features -> env -> train -> evaluate -> report`

### Components
- **Data layer** (`src/rl_riskaware/data/`)
  - CSV ingestion and schema validation.
  - Timestamp ordering and quality checks.
- **Feature layer** (`src/rl_riskaware/features/`)
  - Lag-safe transformations only.
  - Shifted predictors to avoid lookahead leakage.
- **Environment layer** (`src/rl_riskaware/env/`)
  - Gym-style execution environment.
  - Inventory accounting, costs, and terminal completion penalties.
- **Agent layer** (`src/rl_riskaware/agents/`)
  - Gym adapter for policy learning libraries.
  - Policy rollout evaluation helpers.
- **Evaluation layer** (`src/rl_riskaware/evaluation/`)
  - Implementation shortfall and slippage metrics.
  - Walk-forward windowing and grouped summaries.
- **Reporting layer** (`src/rl_riskaware/reporting/`)
  - Reproducible run directories.
  - CSV, markdown, and HTML report outputs with plot artifacts.


## Modeling Assumptions and Controls

The environment currently uses a reduced-form impact/cost specification suitable for controlled experiments:

- participation-driven fill model,
- proportional impact term,
- fixed fee/slippage term,
- terminal penalty for residual inventory.

### Control objectives
- Avoid degenerate zero-trading policies.
- Preserve completion pressure.
- Keep reward magnitudes stable for policy optimization.

For stress scenarios, the framework includes:
- regime-conditioned synthetic series generation,
- multi-seed robustness runs,
- optional staged curriculum controls for difficult regimes.

## Limitations

- Reduced-form impact model (no full LOB dynamics)
- No queue position or matching engine modeling
- Limited to single-asset execution
- Uses proxy features instead of true LOB data

## Quick Start

```bash
git clone https://github.com/minhannscyberspace/RL-RiskAwareExecution.git
cd RL-RiskAwareExecution
pip install -r requirements.txt

python scripts/run_experiment.py --config configs/experiment.default.yaml

## Data Integrity and Leakage Prevention

The platform enforces the following:

- required input schema (`timestamp`, `close`, `volume`),
- monotonic time order and no duplicate timestamps,
- no missing critical fields for execution simulation,
- lag-safe feature construction (predictive terms shifted by one step),
- walk-forward train/test slicing with explicit index boundaries.

No future information is used in state inputs at each decision step.


## Reproducibility

Every major run produces immutable artifact folders under `reports/` with:

- `config_snapshot.yaml`
- key result files (`results.csv`, `summary.csv`, grouped summaries)
- report outputs (`report.md`, `report.html`, plots)

The one-shot pipeline also maintains:

- `reports/latest/LATEST_PIPELINE.txt`
- `reports/latest/LATEST_METADATA.json`

This makes the most recent full run discoverable without searching run history.


## Evaluation 

### Primary metrics
- **Implementation shortfall** (absolute and aggregated)
- **Slippage (bps)**
- **Completion rate**
- **Average execution price**

### Comparative structure
- RL policy vs TWAP/VWAP/POV on aligned windows.
- Walk-forward windows with explicit train/test segmentation.
- Multi-seed summaries to quantify stability.
- Regime-aware breakdowns for behavior under stress.


## Operational Workflow

### Environment setup
```bash
git clone https://github.com/minhannscyberspace/RL-RiskAwareExecution.git
cd RL-RiskAwareExecution

```

### Test gate
```bash
.venv/bin/python -m pytest -q
```

### One-shot experiment run
```bash
MPLCONFIGDIR=.mplconfig MPLBACKEND=Agg .venv/bin/python scripts/run_experiment.py \
  --config configs/experiment.default.yaml
```

This executes train -> eval -> report and writes pipeline metadata.



##  Key Entry Points

- `scripts/train_ppo.py` - train PPO policy.
- `scripts/train_sac.py` - train SAC policy.
- `scripts/eval_ppo_vs_benchmarks.py` - evaluate trained policy vs benchmarks.
- `scripts/run_multiseed_robustness.py` - multi-seed, multi-regime validation.
- `scripts/run_experiment.py` - single-command pipeline execution.
- `scripts/generate_eval_report.py` - report regeneration from an eval folder.


## Repository Structure

- `src/rl_riskaware/` - domain code (data, features, env, agents, evaluation, reporting)
- `configs/` - run configuration templates
- `scripts/` - execution entrypoints
- `tests/` - validation suite
- `reports/` - generated artifacts and summaries

