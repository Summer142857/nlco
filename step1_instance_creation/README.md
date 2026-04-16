# Step 1 Instance Creation

## Overview

Step 1 generates benchmark instances and attaches reference solutions or labels. It is the stage that converts raw benchmark assets or synthetic generation rules into structured JSON files that Step 2 can contextualize.

## Responsibilities

Step 1 is responsible for:

- sampling or loading raw instances
- applying problem-specific extraction logic
- solving or labeling those instances
- writing structured outputs consumed by Step 2

## Key Files and Directories

- `registry.py`: central Step 1 registry
- `problems/`: problem-specific generators and extractors
- `solvers/`: solver wrappers and helper code
- `utils/`: shared generation and data-loading utilities
- `generated_data/`: Step 1 outputs

## Registry Model

The main registration point is [registry.py](registry.py).

Each supported problem has a `ProblemEntry` in `PROBLEM_REGISTRY`. A registry entry defines:

- the problem name
- supported size tiers
- size ranges
- default instance counts
- the generator factory used to produce data

## Inputs

Depending on the problem, Step 1 may use:

- raw assets from `datasets/`
- size ranges and parameters from `problems/<PROBLEM>/config/{S,M,L}.yaml`
- optional problem-specific settings passed through `step1.params`

## Outputs

Generated data is usually written to:

```text
step1_instance_creation/generated_data/<PROBLEM>/
```

Common files include:

- `<PROBLEM>_<SIZE>.json`
- `<PROBLEM>_<SIZE>_stats.json`
- visualization outputs under `viz/` subdirectories for problems that support plotting

## Typical Usage

Step 1 is normally invoked through the top-level pipeline:

```bash
python -m pipeline.run --problem TSP --scale S
```

For direct integration work, the main API surface is the Step 1 registry in `registry.py`.
