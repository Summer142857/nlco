# UI

## Overview

This directory contains the Streamlit interface for the repository.

## Responsibilities

The UI provides a browser-based workflow for common tasks:

- editing problem YAML configs
- running the pipeline
- previewing generated datasets
- editing Step 3 provider keys
- launching evaluation

## Key Files

- `app.py`: Streamlit application entry point
- `runner.py`: helper logic used by the UI to run commands

## Typical Usage

```bash
streamlit run ui/app.py
```

## Files Used by the UI

The UI works on the same files and directories used by the command line:

- `problems/<PROBLEM>/config/{S,M,L}.yaml`
- `step3_evaluation/configs/keys.json`
- `step2_contextualization/dataset/`
- `eval_outputs/`

Before using evaluation features in the UI, create:

```bash
cp step3_evaluation/configs/keys_template.json step3_evaluation/configs/keys.json
```

Then fill the provider keys you need in `step3_evaluation/configs/keys.json`.

Step 2 contextualization can use the same file through its `openai` field, or `OPENAI_API_KEY` from the environment.

Changes made through the UI affect the underlying project files directly.
