# Datasets

## Overview

This directory stores the external data used by the benchmark pipeline. It contains both raw/source assets used by Step 1 and cached contextualization assets reused by Step 2.

## What Lives Here

Examples include:

- graph datasets
- routing datasets
- scheduling datasets
- classical benchmark instance collections
- cached contexts under `cached_context/`

## Expected Layout

Keep the subdirectory structure intact. Many generators refer to fixed relative paths inside `datasets/`.

Examples of existing subdirectories include:

- `CitationNetwork/`
- `JSP/`
- `RCPSP/`
- `ROAD/`
- `RedistrictSet/`
- `STP/`
- `StreetNetwork/`
- `TSP/`
- `cached_context/`

## Download

Populate this folder from the Google Drive data release:

Google Drive: [link](https://drive.google.com/drive/folders/1StvxsrlWw4BVE1YaJW7wWHF7sNdZYpGg?usp=sharing)

After downloading, extract the archive so the content lands in:

```text
<repo-root>/datasets
```

## Notes

- Step 1 usually reads raw benchmark assets from this directory.
- Step 2 may read cached prompt/context files from `datasets/cached_context/`.
- If a generator fails with a missing-path error, the first thing to check is whether the expected dataset subdirectory exists here.
