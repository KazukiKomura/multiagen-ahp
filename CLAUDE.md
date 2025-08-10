# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a research project implementing a **Reason-Based Consensus System (RBCS)** for multi-agent decision-making using Analytic Hierarchy Process (AHP). The system combines Value-based Argumentation Framework (VAF), Argument-Grounded AHP Updates (AGAU), and reinforcement learning (LinUCB) to facilitate consensus formation.

## Core System Architecture

### Main Components

- **VAF (Value-based Argumentation Framework)**: Judges argument acceptance/rejection based on value ordering
- **AHP (Analytic Hierarchy Process)**: Maintains numerical evaluation with weights, scores, and consistency indices  
- **AGAU (Argument-Grounded AHP Updates)**: Updates AHP structure based only on accepted arguments
- **LinUCB Policy**: Learns optimal argument presentation strategies through reinforcement learning

### Key Files

- `rbcs_full_enhanced.py`: Main implementation containing all system components (single-file prototype)
- `rbcs_minimal_prototype.py`: Simplified version for testing
- `runs/`: Directory containing experimental run data with policy files and event logs
- `outputs/`: Log files organized by date and time
- `mlruns/`: MLflow experiment tracking data

## Development Commands

### Running Experiments

**Basic execution (no dependencies):**
```bash
python3 rbcs_full_enhanced.py --episodes 10 --n_agents 5
```

**Full experiment with multiple seeds:**
```bash
for s in 0 1 2 3 4; do 
  RBCS_NO_HYDRA=1 python3 rbcs_full_enhanced.py \
    --task train --mode full --episodes 80 --n_agents 5 --seed $s \
    --ucb_alpha 0.6 --ucb_softmax_temp 0.7 \
    --safety_ci_increase_max 0.03 --safety_tau_drop_max 0.05 \
    --ope_w_clip 5.0 --ope_ridge_lam 5.0 --mlflow
done
```

**Environment variables:**
- `RBCS_NO_HYDRA=1`: Forces argparse fallback (useful when Hydra is not available)
- `HYDRA_AVAILABLE=false`: Alternative way to disable Hydra

### Dependencies

The system is designed to work with minimal dependencies:
- **Required**: `numpy`, `networkx` 
- **Optional**: `hydra-core`, `omegaconf`, `mlflow` (auto-fallback if not available)

Install minimal requirements:
```bash
pip3 install numpy networkx
```

For full functionality:
```bash
pip3 install numpy networkx hydra-core omegaconf mlflow
```

## System Modes

The system supports three experimental modes:

1. **full**: Complete RBCS system with VAF judgment and AGAU updates
2. **explain**: Shows argument acceptance but no AHP updates (explanation-only)
3. **baseline**: Traditional AHP with consistency correction only

## Key Parameters

- `--episodes`: Number of training episodes (default varies by mode)
- `--n_agents`: Number of participant agents (typically 5)
- `--ucb_alpha`: Exploration parameter for LinUCB (0.6 recommended)
- `--safety_*_max`: Safety thresholds for rollback mechanisms
- `--eta_w`, `--eta_S`: Learning rates for weight and score updates
- `--seed`: Random seed for reproducibility

## Output Structure

- **JSONL logs**: `runs/<timestamp>/events.jsonl` - Detailed event tracking
- **Policy files**: `runs/<timestamp>/policy.json` - Learned policy parameters
- **MLflow tracking**: Automatic experiment tracking when available
- **Output logs**: `outputs/<date>/<time>/rbcs_full_enhanced.log`

## Safety Mechanisms

The system includes several safety features:
- **Rollback**: Automatically reverts dangerous updates that violate consistency or fairness thresholds
- **Clipping**: Limits update magnitudes to prevent extreme changes
- **Convergence checks**: Monitors tau (consensus), CI (consistency), and Gini (fairness) metrics

## Research Context

This implementation supports academic research on multi-criteria group decision-making. The system learns optimal argument presentation strategies while maintaining logical consistency and fairness. It's designed for eventual human subject experiments comparing traditional AHP methods with argument-enhanced approaches.

## Code Organization

The current implementation uses a single-file architecture for rapid prototyping. The main classes are:

- `Participant`: Individual agent with weights, scores, and acceptance probabilities
- `LinUCBPolicy`: Learning algorithm for argument strategy selection
- `Environment`: Orchestrates multi-agent interactions and updates
- `JsonlLogger`: Handles structured logging for analysis

## Experimental Design

The system supports within-subjects experiments with three conditions:
- **Baseline**: Traditional AHP approach
- **Explain-only**: Arguments visible but no updates
- **Full**: Complete RBCS with learning and updates

Results are tracked across multiple metrics including convergence time, consistency indices, consensus measures (Kendall-tau), and fairness indicators (Gini coefficient).