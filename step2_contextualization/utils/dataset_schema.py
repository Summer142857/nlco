import csv
import json
from pathlib import Path
from typing import Any, Iterable

csv.field_size_limit(10**8)

V2_FIELDS = [
    "id",
    "task_id",
    "difficulty_tier",
    "example_index",
    "prompt",
    "surface_format",
    "indexing_scheme",
    "instance_canonical_json",
    "reference_solution_canonical_json",
    "reference_objective_value",
    "instance_surface_json",
    "reference_solution_surface_json",
]

INDEXING_SCHEME_MAP = {
    0: "zero_based",
    1: "one_based",
    "0": "zero_based",
    "1": "one_based",
    "zero_based": "zero_based",
    "one_based": "one_based",
    "names": "names",
}


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def infer_task_and_tier_from_base(base_name: str) -> tuple[str | None, str | None]:
    parts = base_name.split("_")
    if len(parts) >= 2 and parts[-1] in {"S", "M", "L"}:
        return "_".join(parts[:-1]), parts[-1]
    return base_name, None


def normalize_record(
    record: dict[str, Any],
    *,
    task_id: str | None = None,
    difficulty_tier: str | None = None,
    example_index: int | None = None,
) -> dict[str, Any]:
    if "id" in record and "task_id" in record and "prompt" in record:
        normalized = {field: record.get(field) for field in V2_FIELDS}
        if normalized["indexing_scheme"] in INDEXING_SCHEME_MAP:
            normalized["indexing_scheme"] = INDEXING_SCHEME_MAP[normalized["indexing_scheme"]]
        return normalized

    task = task_id or record.get("task_name") or record.get("task") or record.get("problem_type")
    tier = difficulty_tier or record.get("difficulty_tier") or record.get("size_bucket")

    row_index = example_index
    if row_index is None:
        row_index = record.get("example_index")
    if row_index is None:
        row_index = record.get("context_index")

    if task is None or tier is None or row_index is None:
        raise ValueError(
            f"Cannot normalize record without task/tier/example_index: "
            f"task={task!r}, tier={tier!r}, example_index={row_index!r}"
        )

    indexing_scheme = INDEXING_SCHEME_MAP.get(record.get("input_index_base"), record.get("input_index_base"))
    if indexing_scheme not in {"zero_based", "one_based", "names"}:
        raise ValueError(f"Unexpected indexing scheme value: {record.get('input_index_base')!r}")

    return {
        "id": f"{task}_{tier}_{int(row_index):03d}",
        "task_id": task,
        "difficulty_tier": tier,
        "example_index": int(row_index),
        "prompt": record.get("instruction", record.get("prompt")),
        "surface_format": record.get("input_format", record.get("surface_format")),
        "indexing_scheme": indexing_scheme,
        "instance_canonical_json": record.get("instance"),
        "reference_solution_canonical_json": record.get("solution"),
        "reference_objective_value": record.get("obj", record.get("objective_value")),
        "instance_surface_json": record.get("instance_variant"),
        "reference_solution_surface_json": record.get("solution_variant"),
    }


def build_output_record(
    *,
    task_id: str,
    difficulty_tier: str,
    example_index: int,
    prompt: str,
    surface_format: str,
    indexing_scheme: Any,
    instance_canonical_json: Any,
    reference_solution_canonical_json: Any,
    reference_objective_value: Any,
    instance_surface_json: Any,
    reference_solution_surface_json: Any,
) -> dict[str, Any]:
    normalized_indexing_scheme = INDEXING_SCHEME_MAP.get(indexing_scheme, indexing_scheme)
    if normalized_indexing_scheme not in {"zero_based", "one_based", "names"}:
        raise ValueError(f"Unexpected indexing scheme value: {indexing_scheme!r}")

    return {
        "id": f"{task_id}_{difficulty_tier}_{int(example_index):03d}",
        "task_id": task_id,
        "difficulty_tier": difficulty_tier,
        "example_index": int(example_index),
        "prompt": prompt,
        "surface_format": surface_format,
        "indexing_scheme": normalized_indexing_scheme,
        "instance_canonical_json": instance_canonical_json,
        "reference_solution_canonical_json": reference_solution_canonical_json,
        "reference_objective_value": reference_objective_value,
        "instance_surface_json": instance_surface_json,
        "reference_solution_surface_json": reference_solution_surface_json,
    }


def normalize_records(
    records: Iterable[dict[str, Any]],
    *,
    task_id: str | None = None,
    difficulty_tier: str | None = None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        normalized.append(
            normalize_record(
                record,
                task_id=task_id,
                difficulty_tier=difficulty_tier,
                example_index=idx,
            )
        )
    return normalized


def records_for_json(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in records:
        out.append(
            {
                "id": record["id"],
                "task_id": record["task_id"],
                "difficulty_tier": record["difficulty_tier"],
                "example_index": record["example_index"],
                "prompt": record["prompt"],
                "surface_format": record["surface_format"],
                "indexing_scheme": record["indexing_scheme"],
                "instance_canonical_json": record["instance_canonical_json"],
                "reference_solution_canonical_json": record["reference_solution_canonical_json"],
                "reference_objective_value": record["reference_objective_value"],
                "instance_surface_json": record["instance_surface_json"],
                "reference_solution_surface_json": record["reference_solution_surface_json"],
            }
        )
    return out


def records_for_csv(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "id": record["id"],
                "task_id": record["task_id"],
                "difficulty_tier": record["difficulty_tier"],
                "example_index": record["example_index"],
                "prompt": record["prompt"],
                "surface_format": record["surface_format"],
                "indexing_scheme": record["indexing_scheme"],
                "instance_canonical_json": _json_dump(record["instance_canonical_json"]),
                "reference_solution_canonical_json": _json_dump(record["reference_solution_canonical_json"]),
                "reference_objective_value": record["reference_objective_value"],
                "instance_surface_json": _json_dump(record["instance_surface_json"]),
                "reference_solution_surface_json": _json_dump(record["reference_solution_surface_json"]),
            }
        )
    return rows


def write_normalized_dataset(records: list[dict[str, Any]], json_path: str | Path, csv_path: str | Path) -> None:
    json_path = Path(json_path)
    csv_path = Path(csv_path)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records_for_json(records), handle, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=V2_FIELDS)
        writer.writeheader()
        writer.writerows(records_for_csv(records))
