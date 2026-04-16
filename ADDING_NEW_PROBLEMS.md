# Adding New Problems

This guide explains how to integrate a new problem `FOO` into the repository.

## Pipeline Flow

For a single problem and scale, the pipeline flow is:

```text
pipeline/run.py
  -> problems/registry.py
  -> problems/_step1_adapter.py
  -> step1_instance_creation/registry.py
  -> problems/_step2_adapter.py
  -> step2_contextualization/problems/_dispatch.py
```

The practical implication is:

- problem discovery is based on `problems/<PROBLEM>/config/`
- Step 1 is selected through `step1_instance_creation/registry.py`
- Step 2 is selected through `step2_contextualization/problems/_dispatch.py`
- Step 3 is added separately if the evaluation logic is new

## Files You Usually Need

For a new problem `FOO`, the common entry points are:

1. `problems/FOO/config/S.yaml`
2. `problems/FOO/config/M.yaml`
3. `problems/FOO/config/L.yaml`
4. `step1_instance_creation/registry.py`
5. `step2_contextualization/problems/_dispatch.py`
6. `step3_evaluation/problem/eval_foo.py`
7. `step3_evaluation/utils/problems_registry.py`
8. `step3_evaluation/utils/objective_map.py`

Not every new problem needs new code in all eight places. That depends on whether it can reuse an existing Step 1 generator, Step 2 contextualizer, and Step 3 evaluator.

## Step 1: Instance Generation

Step 1 is registered in:

- [step1_instance_creation/registry.py](step1_instance_creation/registry.py)

### What Step 1 needs

Step 1 needs a `ProblemEntry` in `PROBLEM_REGISTRY`. A `ProblemEntry` defines:

- the problem name
- supported size tiers
- the size range for each tier
- the default number of instances per tier
- the generator factory used to create instances

If an existing generator already matches your new problem, reuse it. If not, add a new generator implementation under:

```text
step1_instance_creation/problems/
```

### What the Step 1 config must provide

The Step 1 section in `problems/FOO/config/{S,M,L}.yaml` typically contains:

- `output`
- `n_instances`
- a size field such as `n_nodes`, `n_items`, `n_jobs`, `n_tasks`, or another supported size key
- optional dataset location such as `dataset_path` or `dataset_dir`
- optional `params` passed into the Step 1 backend

Example:

```yaml
step1:
  dataset_path: datasets/FOO
  n_nodes: "10-15"
  n_instances: 50
  output: step1_instance_creation/generated_data/FOO/FOO_S.json
  params:
    oversample_factor: 2.0
```

### Step 1 output location

Step 1 usually writes to:

```text
step1_instance_creation/generated_data/FOO/
```

## Step 2: Contextualization

Step 2 dispatch is defined in:

- [step2_contextualization/problems/_dispatch.py](step2_contextualization/problems/_dispatch.py)

### What Step 2 needs

Add `FOO` to `MODULE_BY_PROBLEM` and map it to the contextualization module that should build the natural-language dataset.

Common targets include modules such as:

- `contextualize_graphs`
- `contextualize_sets`
- `contextualize_trees`
- `contextualize_routing`
- `contextualize_packing`
- `contextualize_facility`
- problem-specific modules such as `contextualize_qap`

If none of the existing modules fits, create a new contextualization module under:

```text
step2_contextualization/
```

Current Step 2 modules are expected to emit the normalized benchmark schema directly, rather than first producing an older intermediate schema and converting it later.

### What the Step 2 config must provide

The Step 2 section in `problems/FOO/config/{S,M,L}.yaml` typically contains:

- `instance_dir`
- `output_root_dir`
- `k_per_call`
- `n_target`
- `load_cached_contexts`
- optional cached context path fields, if the module uses them

Example:

```yaml
step2:
  instance_dir: step1_instance_creation/generated_data/FOO
  output_root_dir: step2_contextualization/dataset/FOO
  k_per_call: 20
  n_target: 50
  load_cached_contexts: true
```

### Step 2 output location

Step 2 usually writes to:

```text
step2_contextualization/dataset/FOO/
```

Typical files:

```text
FOO_S_context.csv
FOO_S_context.json
```

The generated rows should follow the normalized Step 2 schema used by the NLCO release, including fields such as:

- `id`
- `task_id`
- `difficulty_tier`
- `example_index`
- `prompt`
- `surface_format`
- `indexing_scheme`
- `instance_canonical_json`
- `reference_solution_canonical_json`
- `reference_objective_value`
- `instance_surface_json`
- `reference_solution_surface_json`

## Step 3: Evaluation

Step 3 is only needed if you want to evaluate model outputs for the new problem.

Relevant files:

- `step3_evaluation/problem/eval_foo.py`
- [step3_evaluation/utils/problems_registry.py](step3_evaluation/utils/problems_registry.py)
- [step3_evaluation/utils/objective_map.py](step3_evaluation/utils/objective_map.py)

### What Step 3 needs

Add Step 3 code when one or more of the following is new:

- the model output format
- the parser for the decision field
- the feasibility check
- the objective computation

If your new problem matches an existing evaluation pattern exactly, reuse the relevant parser and checker instead of creating a new evaluator from scratch.

## Config Directory

Every problem must have:

```text
problems/FOO/config/S.yaml
problems/FOO/config/M.yaml
problems/FOO/config/L.yaml
```

These files are the pipeline entry point for the problem. They define:

- problem name
- scale name
- random seed
- Step 1 settings
- Step 2 settings

## Minimal Integration Checklist

For a problem that reuses existing Step 1 and Step 2 families, the minimum checklist is:

1. Create `problems/FOO/config/S.yaml`
2. Create `problems/FOO/config/M.yaml`
3. Create `problems/FOO/config/L.yaml`
4. Add `FOO` to `step1_instance_creation/registry.py`
5. Add `FOO` to `step2_contextualization/problems/_dispatch.py`

If Step 3 is required, also:

6. Add Step 3 evaluator support

## Validation

### Pipeline smoke test

Run:

```bash
python -m pipeline.run --problem FOO --scale S
```

This validates:

- config loading
- Step 1 registration
- Step 2 registration
- output path wiring

### Step 3 smoke test

If Step 3 support was added, run:

```bash
python -m step3_evaluation.eval_cli \
  --problem FOO \
  --sizes S \
  --dataset_root step2_contextualization/dataset \
  --output_root eval_outputs \
  --model o4-mini \
  --provider openai \
  --reasoning enabled \
  --reasoning_effort high \
  --num_test 2
```
