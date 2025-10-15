# Generative Model Unlearning

## Overview
- Uses task-specific VAEs to replay prior knowledge during continual learning on Split CIFAR-100.
- Covers both incremental training and post-hoc unlearning of selected tasks.
- Historic experiment summaries are stored in `results/`.

## Layout
- `src/generative_model_unlearning/` – package with data preparation, generative models, and runners.
- `results/` – accuracy logs from earlier experiments.
- `requirements.txt` – dependencies shared by training scripts.

## Quickstart
1. `cd repositories/Generative-Continual-Learning-PyTorch`
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `export PYTHONPATH=src`
5. Train with generative replay: `python -m generative_model_unlearning.run_project_c`

### Unlearning demo
```
export PYTHONPATH=src
python -m generative_model_unlearning.generative_unlearning
```

# Dynamic Architecture Continual Learning — Generative Unlearning

One-line: research codebase for continual learning experiments using task-specific generative replay (VAEs) on Split CIFAR-100, including post-hoc unlearning experiments.

This repository contains code used to run and reproduce experiments for generative-model aided continual learning and targeted unlearning of learned tasks.

## Quick links
- Code: `src/generative_model_unlearning/`
- Data: `data/` (CIFAR-100 stored under `data/cifar-100-python/`)
- Results and logs: `results/`
- Python deps: `requirements.txt`

## Requirements
- Python 3.10+ recommended
- PyTorch 2.0+ and torchvision

Install dependencies (recommended inside a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set PYTHONPATH so tests and runners can import the package in `src`:

```bash
export PYTHONPATH=$(pwd)/src
```

## Project structure

- src/generative_model_unlearning/ — core package
	- data_setup.py — CIFAR-100 loading and dataset helpers
	- generative_models.py — VAE and related generator code
	- generative_unlearning.py — scripts to run selective unlearning experiments
	- run_project_c.py — example training/run entrypoint
- data/ — external dataset files (not tracked in git)
- results/ — experiment outputs (not tracked in git)
- requirements.txt — pinned runtime dependencies

If you add new data or large model checkpoints, place them under the `data/` or `results/` directories; these are ignored by git by default.

## Usage examples

Train a model (example):

```bash
# from repository root
export PYTHONPATH=$(pwd)/src
python -m generative_model_unlearning.run_project_c
```

Run the unlearning demo:

```bash
export PYTHONPATH=$(pwd)/src
python -m generative_model_unlearning.generative_unlearning
```

Add or change command-line args in the `if __name__ == '__main__'` blocks of the modules above to customize dataset roots, epochs, or save locations.

## Development

- Follow the Python packaging layout: keep code under `src/` to simplify imports.
- Create a virtual environment (see above).
- Run unit or smoke tests (not currently provided) by creating a `tests/` directory and using `pytest`.

Suggested small improvements you can add:

- Add a `setup.cfg` and `pyproject.toml` for tooling (ruff, black, pytest).
- Add a lightweight test that loads the dataset and runs a single training step to catch API regressions.
