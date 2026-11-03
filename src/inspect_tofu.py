"""Download and inspect the official TOFU forget01 dataset split.

This script only inspects the dataset. It does not load, train, or evaluate a
language model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "locuslab/TOFU"
DATASET_SPLIT = "forget01"
DATASET_REVISION = "324592d84ae4f482ac7249b9285c2ecdb53e3a68"
DATASET_URL = (
    "https://huggingface.co/datasets/"
    f"{DATASET_ID}/resolve/{DATASET_REVISION}/{DATASET_SPLIT}.json?download=true"
)
DATASET_SHA256 = "d0047332bd27a489c552659831f4ffff080e2b9b6e063c7c951ac4de9d494a63"
DEFAULT_DATA_PATH = PROJECT_ROOT / ".cache" / "tofu" / "forget01.json"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "results" / "tofu_forget01_inspection.json"
REQUIRED_FIELDS = ("question", "answer")


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as data_file:
        for chunk in iter(lambda: data_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(destination: Path = DEFAULT_DATA_PATH) -> Path:
    """Fetch the pinned official split and verify it before saving it locally."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == DATASET_SHA256:
        return destination

    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "UnlearnGuard-TOFU-inspection/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Unable to download the official TOFU split from {DATASET_URL}: {exc}"
        ) from exc

    digest = hashlib.sha256(content).hexdigest()
    if digest != DATASET_SHA256:
        raise RuntimeError(
            "The downloaded file checksum does not match the pinned TOFU revision "
            f"(expected {DATASET_SHA256}, got {digest})."
        )

    destination.write_bytes(content)
    return destination


def load_jsonl(path: Path) -> list[Any]:
    """Load one JSON object per nonblank line (the Hub file uses JSONL)."""

    records: list[Any] = []
    with path.open("r", encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc.msg}"
                ) from exc
    return records


def inspect_records(records: list[Any]) -> dict[str, Any]:
    """Discover the record schema, then validate TOFU's required QA fields."""

    if not records:
        raise ValueError("The dataset contains no records.")

    types_by_field: dict[str, set[str]] = defaultdict(set)
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be a JSON object.")
        for field, value in record.items():
            types_by_field[field].add(type(value).__name__)

    # Determine the actual schema above before checking the expected QA fields.
    discovered_fields = set(types_by_field)
    missing_schema_fields = sorted(set(REQUIRED_FIELDS) - discovered_fields)
    if missing_schema_fields:
        raise ValueError(
            "Dataset schema is missing required field(s): "
            + ", ".join(missing_schema_fields)
            + f". Discovered fields: {', '.join(sorted(discovered_fields)) or '(none)'}"
        )

    for index, record in enumerate(records, start=1):
        for field in REQUIRED_FIELDS:
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Record {index} has a missing, empty, or non-string {field!r} field."
                )

    return {
        "record_count": len(records),
        "schema": {
            field: sorted(type_names)
            for field, type_names in sorted(types_by_field.items())
        },
        "required_fields": list(REQUIRED_FIELDS),
        "required_fields_valid": True,
    }


def build_summary(
    inspection: dict[str, Any],
    *,
    file_sha256: str,
    source: str,
    revision: str | None,
) -> dict[str, Any]:
    """Create a small metadata-only summary; never copy question/answer text."""

    return {
        "dataset": DATASET_ID,
        "split": DATASET_SPLIT,
        "source": source,
        "revision": revision,
        "file_sha256": file_sha256,
        "record_count": inspection["record_count"],
        "schema": inspection["schema"],
        "required_fields": inspection["required_fields"],
        "required_fields_valid": inspection["required_fields_valid"],
        "representative_examples_displayed": 3,
        "record_text_included": False,
    }


def representative_indices(record_count: int) -> list[int]:
    """Return three stable, zero-based positions spread across the data."""

    if record_count < 3:
        raise ValueError("At least three records are required to display examples.")
    return [0, (record_count - 1) // 2, record_count - 1]


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def display_inspection(
    records: list[dict[str, Any]], inspection: dict[str, Any], *, revision: str | None
) -> None:
    print(f"Dataset: {DATASET_ID} / {DATASET_SPLIT}")
    if revision:
        print(f"Pinned revision: {revision}")
    print(f"Record count: {inspection['record_count']}")
    print("Discovered schema (field: observed JSON value type):")
    for field, types in inspection["schema"].items():
        print(f"  - {field}: {', '.join(types)}")
    print(
        "Required fields validated: "
        + ", ".join(inspection["required_fields"])
    )
    print("\nRepresentative fictional question-answer examples:")
    for example_number, index in enumerate(
        representative_indices(len(records)), start=1
    ):
        record = records[index]
        print(f"\nExample {example_number} (record {index + 1}):")
        print(f"Q: {record['question']}")
        print(f"A: {record['answer']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        type=Path,
        help="Inspect a local JSONL file instead of downloading the pinned official split.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Metadata-only summary output (default: {DEFAULT_SUMMARY_PATH}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.data_file is None:
        data_path = download_dataset()
        source = DATASET_URL
        revision: str | None = DATASET_REVISION
    else:
        data_path = args.data_file
        source = "local file supplied with --data-file"
        revision = None

    records = load_jsonl(data_path)
    inspection = inspect_records(records)
    summary = build_summary(
        inspection,
        file_sha256=sha256_file(data_path),
        source=source,
        revision=revision,
    )
    write_summary(args.summary_file, summary)
    display_inspection(records, inspection, revision=revision)
    print(f"\nMetadata-only summary saved to: {args.summary_file}")
    print("This inspection does not train, evaluate, or unlearn a model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
