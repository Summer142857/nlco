# Problem Data Sources And Instance Scales

This table is derived from `problems/*/config/{S,M,L}.yaml`.

- `Data source` comes from `step1.dataset_dir` or `step1.dataset_path`.
- If neither field exists, the value is `generated`.

| Problem | Data source | S | M | L |
|---|---|---|---|---|
| 2SP | `datasets/2SP` | `n_items: 5-10` | `n_items: 11-15` | `n_items: 16-25` |
| AP3 | `datasets/AP3` | `n_nodes: 3-5` | `n_nodes: 6-8` | `n_nodes: 9-12` |
| BPP | `datasets/BPP/` | `n_nodes: 5-10` | `n_nodes: 11-15` | `n_nodes: 16-25` |
| CFLP | `datasets/CFLP` | `n_nodes: 5-8` | `n_nodes: 9-14` | `n_nodes: 15-20` |
| CMP | `datasets/CMP` | `n_nodes: 8-12` | `n_nodes: 13-18` | `n_nodes: 19-25` |
| CSP | `datasets/CSP` | `n_types: 5-10` | `n_types: 11-15` | `n_types: 16-25` |
| CVRP | `datasets/CVRP` | `n_nodes: 5-10` | `n_nodes: 11-15` | `n_nodes: 16-25` |
| FSP | `generated` | `n_jobs: 2-5; n_machines: 2-3` | `n_jobs: 6-8; n_machines: 3-5` | `n_jobs: 9-12; n_machines: 5-6` |
| GAP | `datasets/GAP` | `n_tasks: 5-10` | `n_tasks: 11-15` | `n_tasks: 16-25` |
| GCP | `datasets/CitationNetwork` | `n_nodes: 8-12` | `n_nodes: 13-18` | `n_nodes: 19-25` |
| HSP | `datasets/ROAD` | `n_nodes: 8-12` | `n_nodes: 13-18` | `n_nodes: 19-25` |
| JSP | `generated` | `n_jobs: 2-5; n_machines: 2-3` | `n_jobs: 6-8; n_machines: 3-5` | `n_jobs: 9-12; n_machines: 5-6` |
| KMST | `datasets/STP` | `n_nodes: 10-15` | `n_nodes: 16-20` | `n_nodes: 21-30` |
| KP | `generated` | `n_items: 5-10` | `n_items: 11-20` | `n_items: 21-30` |
| LOP | `datasets/LOP` | `n_nodes: 5-10` | `n_nodes: 11-15` | `n_nodes: 16-25` |
| MAXCUT | `datasets/RedistrictSet` | `n_nodes: 8-12` | `n_nodes: 13-18` | `n_nodes: 19-25` |
| MCP | `datasets/CitationNetwork` | `n_nodes: 8-12` | `n_nodes: 13-18` | `n_nodes: 19-25` |
| MDP | `datasets/MDP` | `n_vertices: 8-12` | `n_vertices: 13-18` | `n_vertices: 19-25` |
| MDS | `datasets/StreetNetwork` | `n_nodes: 8-12` | `n_nodes: 13-18` | `n_nodes: 19-25` |
| MIS | `datasets/RedistrictSet` | `n_nodes: 8-12` | `n_nodes: 13-18` | `n_nodes: 19-25` |
| MLP | `datasets/TSP` | `n_nodes: 5-10` | `n_nodes: 11-15` | `n_nodes: 16-25` |
| MVC | `datasets/RedistrictSet` | `n_nodes: 8-12` | `n_nodes: 13-18` | `n_nodes: 19-25` |
| MkC | `datasets/ROAD` | `n_nodes: 8-12` | `n_nodes: 13-18` | `n_nodes: 19-25` |
| OP | `datasets/OP` | `n_nodes: 5-10` | `n_nodes: 11-15` | `n_nodes: 16-25` |
| OSP | `generated` | `n_jobs: 2-5; n_machines: 2-3` | `n_jobs: 6-8; n_machines: 3-5` | `n_jobs: 9-12; n_machines: 5-6` |
| PCENTER | `datasets/cached_context1/PMED` | `n_vertices: 5-8` | `n_vertices: 9-14` | `n_vertices: 15-20` |
| PCTSP | `datasets/TSP/TSPlib_70instances.txt` | `n_nodes: 5-10` | `n_nodes: 11-15` | `n_nodes: 16-25` |
| PDP | `datasets/PDP` | `n_requests: 3-5` | `n_requests: 6-8` | `n_requests: 9-14` |
| PMED | `datasets/cached_context1/PMED` | `n_vertices: 5-8` | `n_vertices: 9-14` | `n_vertices: 15-20` |
| PMS | `datasets/PMS` | `n_jobs: 5-10; n_machines: 2` | `n_jobs: 11-15; n_machines: 2-3` | `n_jobs: 16-25; n_machines: 2-5` |
| QAP | `datasets/QAP` | `n_nodes: 3-5` | `n_nodes: 6-8` | `n_nodes: 9-12` |
| QKP | `datasets/QKP` | `n_items: 5-10` | `n_items: 11-15` | `n_items: 16-25` |
| QSPP | `generated` | `n_nodes: 5-10` | `n_nodes: 11-15` | `n_nodes: 16-25` |
| RCPSP | `datasets/RCPSP` | `n_tasks: 4-10` | `n_tasks: 11-15` | `n_tasks: 16-21` |
| SCP | `datasets/ROAD` | `n_nodes: 8-12` | `n_nodes: 13-18` | `n_nodes: 19-25` |
| SFP | `datasets/STP` | `n_nodes: 10-15` | `n_nodes: 16-20` | `n_nodes: 21-30` |
| SMTWT | `generated` | `n_jobs: 5-10` | `n_jobs: 11-20` | `n_jobs: 21-30` |
| SP | `datasets/ROAD` | `n_nodes: 8-12` | `n_nodes: 13-18` | `n_nodes: 19-25` |
| SPP | `generated` | `items_range: 8-12` | `items_range: 13-18` | `items_range: 19-25` |
| STP | `datasets/STP` | `n_nodes: 10-15` | `n_nodes: 16-20` | `n_nodes: 21-30` |
| TSP | `datasets/TSP/TSPlib_70instances.txt` | `n_nodes: 5-10` | `n_nodes: 11-15` | `n_nodes: 16-25` |
| TSPTW | `datasets/TSP` | `n_nodes: 5-10` | `n_nodes: 11-15` | `n_nodes: 16-25` |
| UFLP | `datasets/UFLP` | `n_nodes: 5-8` | `n_nodes: 9-14` | `n_nodes: 15-20` |
