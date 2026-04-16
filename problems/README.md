# Problems

This directory contains the per-problem YAML configuration used by the pipeline, plus the small registry/adaptor layer that routes one selected problem into Step 1 and Step 2.

## Purpose

This folder provides:

- one config directory per benchmark task,
- one YAML file per difficulty tier (`S`, `M`, `L`),
- a registry that discovers available problems,
- adapters that forward YAML fields into the Step 1 and Step 2 implementations.

## Layout

```text
problems/
  registry.py
  _step1_adapter.py
  _step2_adapter.py
  <PROBLEM>/
    config/
      S.yaml
      M.yaml
      L.yaml
```

## How Config Resolution Works

When the pipeline receives:

```bash
python -m pipeline.run --problem TSP --scale S
```

it does the following:

1. loads `problems/TSP/config/S.yaml`
2. resolves the problem entry through `registry.py`
3. sends the parsed config to `_step1_adapter.py`
4. sends the same config to `_step2_adapter.py`
5. Step 1 and Step 2 read only the fields they need from the YAML

## YAML Structure

Each problem config follows the same high-level shape:

```yaml
problem: TSP
scale: S
seed: 43

step1:
  dataset_dir: datasets/TSP/TSPlib_70instances.txt
  n_nodes: 5-10
  n_instances: 50
  output: step1_instance_creation/generated_data/TSP/TSP_S.json

step2:
  load_cached_contexts: true
  contexts_audit_path: datasets/cached_context/TSP/TSP_contexts.json
  instance_dir: step1_instance_creation/generated_data/TSP
  output_root_dir: step2_contextualization/dataset/TSP
  k_per_call: 20
  n_target: 30
```

## Root-Level Fields

These fields appear at the top of almost every config:

| field | meaning |
|---|---|
| `problem` | Benchmark task code, for example `TSP`, `BPP`, `RCPSP` |
| `scale` | Difficulty tier: `S`, `M`, or `L` |
| `seed` | Global random seed used by the pipeline or passed into Step 1 generators |

## `step1` Section

The `step1` block controls raw instance generation and solver annotation.

### Common `step1` fields

| field | meaning |
|---|---|
| `output` | Output JSON path for Step 1 generated instances |
| `n_instances` | Number of instances Step 1 should generate for this problem-tier |
| `seed` | Optional Step 1-specific seed override |
| `dataset_dir` | Input directory or file used by the Step 1 generator |
| `dataset_path` | Alternative to `dataset_dir`; used by some generators that expect a path field with this exact name |
| `params` | Problem-specific extra parameters forwarded to the underlying Step 1 implementation |

### Range-style size controls

Most problems express difficulty through one or more range fields. These are usually strings like `5-10`, meaning Step 1 should sample values inside that interval.

Common examples:

| field | typical use |
|---|---|
| `n_nodes` | routing / graph / tree problems |
| `n_vertices` | facility / graph variants that use vertex terminology |
| `n_items` | packing / knapsack problems |
| `items_range` | set packing / set partitioning style tasks |
| `n_jobs` | scheduling problems |
| `n_tasks` | project scheduling or task-oriented generators |
| `n_requests` | pickup-and-delivery / routing variants |
| `n_types` | typed item generators |

These fields are not globally required. Each Step 1 generator only reads the subset relevant to that problem.

### `step1.params`

`params` is the escape hatch for problem-specific knobs that do not fit into the shared schema.  
Examples found in the current configs include:

- graph sampling controls such as `min_density`, `max_density`, `alpha`, `steps`
- oversampling and candidate filtering controls such as `oversample_factor`
- scheduling controls such as `n_machines`
- solver/runtime settings such as `time_limit`, `threads`
- tree/set generation controls such as `k_ratio_choices`, `t_ratio_choices`, `min_groups`, `max_groups`
- visualization output locations such as `viz_dir`

Because `params` is passed into the corresponding Step 1 problem implementation, the exact semantics are problem-specific. When editing `params`, check the matching generator under `step1_instance_creation/problems/`.

## `step2` Section

The `step2` block controls natural-language contextualization and final dataset writing.

### Shared `step2` fields

| field | meaning |
|---|---|
| `instance_dir` | Directory containing Step 1 output JSON files |
| `output_root_dir` | Directory where Step 2 writes contextualized dataset files |
| `n_target` | Number of Step 1 instances to contextualize from the source file |
| `k_per_call` | Number of contexts/templates requested per LLM call during context generation |
| `load_cached_contexts` | Whether Step 2 should reuse previously generated contexts instead of regenerating them |
| `contexts_audit_path` | Path to the cached context JSON audit file |

### `step2` behavior notes

- `instance_dir` usually points to `step1_instance_creation/generated_data/<PROBLEM>`
- `output_root_dir` usually points to `step2_contextualization/dataset/<PROBLEM>`
- if `load_cached_contexts: true` and `contexts_audit_path` exists, Step 2 reuses those contexts
- if `load_cached_contexts: false`, Step 2 regenerates contexts even if a cache file exists
- `n_target` can be smaller than `step1.n_instances`; this is common when Step 1 generates more raw instances than Step 2 currently contextualizes

## Practical Editing Guidance

### If you want to change benchmark size

Edit the range fields in `step1`, for example:

- `n_nodes`
- `n_items`
- `n_jobs`
- `n_tasks`

### If you want to change how many examples are generated

Edit:

- `step1.n_instances` to change how many raw instances are created
- `step2.n_target` to change how many of those instances are contextualized

### If you want to redirect outputs

Edit:

- `step1.output`
- `step2.output_root_dir`

### If you want to reuse or regenerate contexts

Edit:

- `step2.load_cached_contexts`
- `step2.contexts_audit_path`

## Typical Patterns Seen in Existing Configs

### File or directory-backed generators

Many tasks use:

- `dataset_dir`
- `dataset_path`

to point at raw source assets under `datasets/`.

### Pure synthetic generators

Some tasks do not need external datasets and instead rely on size ranges plus `params`. These typically use fields like:

- `n_jobs`
- `n_items`
- `items_range`
- `params.n_machines`

### Hybrid generators

Some problems sample subinstances from raw benchmark libraries and also require extra shaping parameters in `params`, for example density thresholds, oversampling ratios, or graph expansion settings.

## Main Files

- `registry.py`: discovers available problems and returns pipeline callables
- `_step1_adapter.py`: forwards YAML fields into the Step 1 registry interface
- `_step2_adapter.py`: forwards the selected problem and config into Step 2 dispatch

## Adding a Problem

To add a new problem:

1. create `problems/<PROBLEM>/config/`
2. add `S.yaml`, `M.yaml`, and `L.yaml`
3. make sure Step 1 can consume the `step1` fields you define
4. make sure Step 2 dispatch supports the problem code

Detailed instructions:

- [ADDING_NEW_PROBLEMS.md](../ADDING_NEW_PROBLEMS.md)
