# Evaluation Run Commands

This file collects practical command templates for running `step3_evaluation.eval_cli` across providers and models.

## Conventions

- `--reasoning disabled` means no explicit reasoning / thinking mode
- `--reasoning enabled` means reasoning / thinking mode is enabled
- `--reasoning_effort {low|medium|high}` is only valid when `--reasoning enabled`
- `--anthropic_budget` is only used for Anthropic when `--reasoning enabled`

## Common Paths

Use repository-relative paths:

```bash
DATASET_ROOT=step2_contextualization/dataset
OUTPUT_ROOT=eval_outputs
SIZES=S,M,L
```

If you are not in the repository root, replace them with absolute paths for your local checkout.

## API Key Setup

Create a local key file from the template before running evaluation:

```bash
cp step3_evaluation/configs/keys_template.json step3_evaluation/configs/keys.json
```

Then edit `step3_evaluation/configs/keys.json` and fill the provider fields you need.

Step 2 contextualization can also reuse the same file through the `openai` field, or you can set:

```bash
export OPENAI_API_KEY=your_openai_key
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = "your_openai_key"
```

---

## OpenRouter

### `qwen3-14b` without reasoning

```bash
python -m step3_evaluation.eval_cli \
  --problem all \
  --sizes $SIZES \
  --dataset_root $DATASET_ROOT \
  --output_root $OUTPUT_ROOT \
  --model qwen3-14b \
  --provider openrouter \
  --reasoning disabled \
  --max_tokens 16384 
```

### `qwen3-14b` with reasoning

```bash
python -m step3_evaluation.eval_cli \
  --problem all \
  --sizes $SIZES \
  --dataset_root $DATASET_ROOT \
  --output_root $OUTPUT_ROOT \
  --model qwen3-14b \
  --provider openrouter \
  --reasoning enabled \
  --reasoning_effort high
```

### `qwq-32b` with reasoning

```bash
python -m step3_evaluation.eval_cli \
  --problem all \
  --sizes $SIZES \
  --dataset_root $DATASET_ROOT \
  --output_root $OUTPUT_ROOT \
  --model qwq-32b \
  --provider openrouter \
  --reasoning enabled \
  --reasoning_effort high
```

### `ministral-14b-2512` without reasoning

```bash
python -m step3_evaluation.eval_cli \
  --problem all \
  --sizes $SIZES \
  --dataset_root $DATASET_ROOT \
  --output_root $OUTPUT_ROOT \
  --model ministral-14b-2512 \
  --provider openrouter \
  --reasoning disabled \
  --max_tokens 65536 
```

### `grok-4.1-fast` with reasoning

```bash
python -m step3_evaluation.eval_cli \
  --problem all \
  --sizes $SIZES \
  --dataset_root $DATASET_ROOT \
  --output_root $OUTPUT_ROOT \
  --model grok-4.1-fast \
  --provider openrouter \
  --reasoning enabled \
  --reasoning_effort high
```

### `nemotron-3-nano-30b-a3b` with reasoning

```bash
python -m step3_evaluation.eval_cli \
  --problem all \
  --sizes $SIZES \
  --dataset_root $DATASET_ROOT \
  --output_root $OUTPUT_ROOT \
  --model nemotron-3-nano-30b-a3b \
  --provider openrouter \
  --reasoning enabled \
  --reasoning_effort high \
  --max_concurrency 50
```

### `nemotron-3-nano-30b-a3b` without reasoning

```bash
python -m step3_evaluation.eval_cli \
  --problem all \
  --sizes $SIZES \
  --dataset_root $DATASET_ROOT \
  --output_root $OUTPUT_ROOT \
  --model nemotron-3-nano-30b-a3b \
  --provider openrouter \
  --reasoning disabled \
  --max_tokens 65536 \
  --max_concurrency 50
```

---

## Gemini

### `gemini-3-flash-preview` with reasoning

```bash
python -m step3_evaluation.eval_cli \
  --problem all \
  --sizes $SIZES \
  --dataset_root $DATASET_ROOT \
  --output_root $OUTPUT_ROOT \
  --model gemini-3-flash-preview \
  --provider gemini \
  --reasoning enabled \
  --reasoning_effort high
```

---

## Anthropic

### `claude-sonnet-4-5` without reasoning

```bash
python -m step3_evaluation.eval_cli \
  --problem all \
  --sizes $SIZES \
  --dataset_root $DATASET_ROOT \
  --output_root $OUTPUT_ROOT \
  --model claude-sonnet-4-5 \
  --provider anthropic \
  --reasoning disabled \
  --max_tokens 64000
```

### `claude-sonnet-4-5` with reasoning

```bash
python -m step3_evaluation.eval_cli \
  --problem all \
  --sizes $SIZES \
  --dataset_root $DATASET_ROOT \
  --output_root $OUTPUT_ROOT \
  --model claude-sonnet-4-5 \
  --provider anthropic \
  --reasoning enabled \
  --anthropic_budget 60000 \
  --max_tokens 64000
```

---

## Vertex MaaS

### `llama-4-maverick-17b-128e-instruct-maas` without reasoning

```bash
python -m step3_evaluation.eval_cli \
  --problem all \
  --sizes $SIZES \
  --dataset_root $DATASET_ROOT \
  --output_root $OUTPUT_ROOT \
  --model llama-4-maverick-17b-128e-instruct-maas \
  --provider vertex_maas \
  --reasoning disabled \
  --max_tokens 8192
```

### `qwen3-235b-a22b-instruct-2507-maas` without reasoning

```bash
python -m step3_evaluation.eval_cli \
  --problem all \
  --sizes $SIZES \
  --dataset_root $DATASET_ROOT \
  --output_root $OUTPUT_ROOT \
  --model qwen3-235b-a22b-instruct-2507-maas \
  --provider vertex_maas \
  --reasoning disabled \
  --max_tokens 16384
```

---

## DeepSeek

### `deepseek-reasoner` with reasoning

```bash
python -m step3_evaluation.eval_cli \
  --problem all \
  --sizes $SIZES \
  --dataset_root $DATASET_ROOT \
  --output_root $OUTPUT_ROOT \
  --model deepseek-reasoner \
  --provider deepseek \
  --reasoning enabled \
  --max_concurrency 50
```

### `deepseek-chat` without reasoning

```bash
python -m step3_evaluation.eval_cli \
  --problem all \
  --sizes $SIZES \
  --dataset_root $DATASET_ROOT \
  --output_root $OUTPUT_ROOT \
  --model deepseek-chat \
  --provider deepseek \
  --reasoning disabled \
  --max_tokens 8192 \
  --max_concurrency 50
```

---

## OpenAI

### `o4-mini` with reasoning

```bash
python -m step3_evaluation.eval_cli \
  --problem all \
  --sizes $SIZES \
  --dataset_root $DATASET_ROOT \
  --output_root $OUTPUT_ROOT \
  --model o4-mini \
  --provider openai \
  --reasoning enabled \
  --reasoning_effort high 
```

### `gpt-5.1` with reasoning

```bash
python -m step3_evaluation.eval_cli \
  --problem all \
  --sizes $SIZES \
  --dataset_root $DATASET_ROOT \
  --output_root $OUTPUT_ROOT \
  --model gpt-5.1 \
  --provider openai \
  --reasoning enabled \
  --reasoning_effort medium
```

---

## Notes

- These are example command templates, not an exhaustive list of supported models.
- Supported providers in the current CLI include `openai`, `anthropic`, `deepseek`, `gemini`, `vertex_maas`, `openrouter`, and `xiaomi`.
- Dataset files are expected under `step2_contextualization/dataset/<PROBLEM>/`.
- Evaluation outputs are written under `eval_outputs/<PROBLEM>/<MODEL>/`.
- Provider credentials are read from `step3_evaluation/configs/keys.json`.
