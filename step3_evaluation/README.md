# Step 3 Evaluation

## Overview

Step 3 evaluates model outputs on the contextualized datasets produced by Step 2. It runs provider calls, parses returned solutions, checks feasibility, computes objective quality, and writes evaluation tables.

## Responsibilities

Step 3 is responsible for:

- reading contextualized benchmark rows
- calling the selected model provider
- parsing model outputs
- checking feasibility and scoring objective values
- writing per-size and overall evaluation tables

## Key Files and Directories

- `eval_cli.py`: command-line evaluation entry point
- `utils/`: provider calls, parsing, metrics, and evaluation utilities
- `problem/`: problem-specific evaluation logic
- `configs/`: API keys and evaluation-side configuration files

## Inputs

Default input root:

```text
step2_contextualization/dataset
```

Evaluation API keys are read from:

```text
step3_evaluation/configs/keys.json
```

Create it from the template first:

```bash
cp step3_evaluation/configs/keys_template.json step3_evaluation/configs/keys.json
```

Then fill only the providers you plan to use.

Expected key fields:

```json
{
  "openai": "...",
  "anthropic": "...",
  "deepseek": "...",
  "gemini": "...",
  "openrouter": "...",
  "xiaomi": "...",
  "vertex_project_id": "..."
}
```

## Outputs

Default output root:

```text
eval_outputs
```

Per-size output files:

```text
<output_root>/<PROBLEM>/<MODEL>/<PROBLEM>_<SIZE>_eval.csv
```

Overall summary files:

```text
<output_root>/<PROBLEM>/<MODEL>/<PROBLEM>_<MODEL>_overall.csv
```

## Typical Usage

Basic command:

```bash
python -m step3_evaluation.eval_cli \
  --problem TSP \
  --sizes S \
  --dataset_root step2_contextualization/dataset \
  --output_root eval_outputs \
  --model o4-mini \
  --provider openai \
  --reasoning enabled \
  --reasoning_effort high
```

## Provider Notes

OpenAI, Gemini, OpenRouter, and Xiaomi:

```bash
--reasoning enabled --reasoning_effort low|medium|high
```

Anthropic:

```bash
--reasoning enabled --anthropic_budget N
```

DeepSeek:

- use `--model deepseek-reasoner` for reasoning mode
- use `--model deepseek-chat` for non-reasoning mode

## Useful Options

- `--problem TSP,CVRP,MIS` or `--problem all`
- `--sizes S`, `--sizes S,M`, or `--sizes all`
- `--max_concurrency 50`
- `--num_test 5`
- `--decision_field solution`

## Notes

- Existing evaluation CSVs are resumed and locally re-evaluated when possible.
- Evaluation in this repository is online evaluation through provider APIs.
- The Streamlit UI uses the same command-line interface and key file.
