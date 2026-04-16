# Step 2 Contextualization

## Overview

Step 2 converts structured Step 1 outputs into natural-language benchmark datasets. It is the stage that turns solver-annotated JSON instances into prompts and normalized benchmark rows.

## Responsibilities

Step 2 is responsible for:

- reading structured instances from Step 1
- building scenario text and prompt variants
- writing normalized CSV and JSON benchmark datasets
- saving final files per problem and difficulty tier

## Key Files and Directories

- `problems/_dispatch.py`: central problem-to-contextualizer mapping
- `contextualize_*.py`: problem-specific contextualization modules
- `utils/`: prompt building, scenario generation, and dataset helpers
- `dataset/`: Step 2 outputs

## Inputs

Step 2 reads:

- Step 1 outputs from `step1_instance_creation/generated_data/<PROBLEM>/`
- Step 2 config fields from `problems/<PROBLEM>/config/{S,M,L}.yaml`
- optional cached contexts from `datasets/cached_context/`

Typical Step 2 config fields:

- `instance_dir`
- `output_root_dir`
- `k_per_call`
- `n_target`
- `load_cached_contexts`
- `contexts_audit_path`

## API Key Setup

Step 2 may need an OpenAI API key when it regenerates contexts with `load_cached_contexts: false`.

It looks for the key in this order:

1. `OPENAI_API_KEY`
2. `step3_evaluation/configs/keys.json` under the `openai` field

Recommended setup:

```bash
cp step3_evaluation/configs/keys_template.json step3_evaluation/configs/keys.json
```

Then fill the `openai` field, or set an environment variable instead:

```bash
export OPENAI_API_KEY=your_openai_key
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = "your_openai_key"
```

When `load_cached_contexts: true` and a valid `contexts_audit_path` is available, Step 2 can reuse cached contexts instead of generating a new context pool.

## Outputs

Step 2 writes normalized benchmark data to:

```text
step2_contextualization/dataset/<PROBLEM>/
```

Typical files:

- `<PROBLEM>_<SIZE>_context.csv`
- `<PROBLEM>_<SIZE>_context.json`

## Dataset Schema

The CSV/JSON record schema is aligned with the Hugging Face NLCO release:

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

## Typical Usage

Step 2 is normally run through the top-level pipeline:

```bash
python -m pipeline.run --problem TSP --scale S
```

If you add a new problem, register it in `problems/_dispatch.py` and map it to the correct contextualization module.
