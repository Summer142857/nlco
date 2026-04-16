# Pipeline

## Overview

This directory contains the orchestration entry points that run Step 1 and Step 2 for one selected problem configuration. The pipeline layer loads YAML configs, resolves the selected problem through the registry, and forwards the config into the Step 1 and Step 2 adapters.

The pipeline does not contain problem-specific generation or evaluation logic.

## Key Files

- `run.py`: run one problem at one scale
- `run_all.py`: run one or more problems across one or more scales
- `config_loader.py`: load and merge YAML configs

## Inputs

The pipeline expects:

- a config file under `problems/<PROBLEM>/config/{S,M,L}.yaml`
- raw data under `datasets/` when the selected problem requires external assets

## Outputs

Pipeline execution writes:

- Step 1 outputs under `step1_instance_creation/generated_data/<PROBLEM>/`
- Step 2 outputs under `step2_contextualization/dataset/<PROBLEM>/`

## Typical Usage

Run one problem and one scale:

```bash
python -m pipeline.run --problem TSP --scale S
```

Run multiple problems:

```bash
python -m pipeline.run_all --scale S
python -m pipeline.run_all --scale S --only TSP
python -m pipeline.run_all --scales S,M
```

## Execution Flow

`run.py` follows this path:

```text
config YAML
  -> problems/registry.py
  -> problems/_step1_adapter.py
  -> step1_instance_creation/registry.py
  -> problems/_step2_adapter.py
  -> step2_contextualization/problems/_dispatch.py
```
