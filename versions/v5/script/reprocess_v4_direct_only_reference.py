from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
VERSION_DIR = SCRIPT_DIR.parent
PROJECT_DIR = VERSION_DIR.parent.parent
V4_EXTRACTION_DIR = PROJECT_DIR / "versions" / "v4" / "extraction"
V5_EXTRACTION_DIR = VERSION_DIR / "extraction"

POLICY_VERSION = "v5.0-direct-evidence-only"

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

STATUS_CODES = {
    "P": "explicit_positive",
    "N": "explicit_negative",
    "U": "indeterminate",
    "M": "not_mentioned",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build v5 from the existing v4 direct LLM extractions. "
            "V5 keeps every direct v4 judgment and removes all related-finding supplementation."
        )
    )
    parser.add_argument(
        "--source-jsonl",
        default=str(V4_EXTRACTION_DIR / "gemma4_e4b_extract_58_v4.jsonl"),
        help="Existing v4 JSONL containing the direct LLM extractions.",
    )
    parser.add_argument(
        "--jsonl",
        default=str(V5_EXTRACTION_DIR / "gemma4_e4b_extract_58_v5.jsonl"),
    )
    parser.add_argument(
        "--long-csv",
        default=str(V5_EXTRACTION_DIR / "gemma4_e4b_extract_58_v5_long.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        default=str(V5_EXTRACTION_DIR / "gemma4_e4b_extract_58_v5_summary.csv"),
    )
    return parser.parse_args()


def validate_direct_extraction(extraction: dict[str, Any]) -> None:
    if set(extraction.get("targets", {})) != set(TARGETS):
        raise ValueError("The source extraction does not contain exactly the 12 targets.")

    for target in TARGETS:
        item = extraction["targets"][target]
        code = item.get("s")
        evidence = str(item.get("e", "")).strip()
        if code not in STATUS_CODES:
            raise ValueError(f"Invalid status code for {target}: {code}")
        if code == "M" and evidence:
            raise ValueError(f"Not-mentioned target has direct evidence for {target}: {evidence}")
        if code != "M" and not evidence:
            raise ValueError(f"Status {code} requires direct evidence for {target}")


def apply_v5_policy(extraction: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Keep direct judgments only; never supplement one target from another finding."""
    validate_direct_extraction(extraction)
    processed: dict[str, dict[str, Any]] = {}

    for target in TARGETS:
        item = extraction["targets"][target]
        direct_code = item["s"]
        direct_status = STATUS_CODES[direct_code]
        direct_binary = 1 if direct_code == "P" else 0 if direct_code == "N" else None
        processed[target] = {
            "direct_status": direct_status,
            "direct_binary": direct_binary,
            "direct_evidence": str(item.get("e", "")).strip(),
            "final_status": direct_status,
            "final_binary": direct_binary,
            "inference_source": "direct" if direct_code != "M" else "",
            "inference_rule": "direct_report_only",
            "related_evidence": "",
        }

    return processed


def comparison_category(gold: int, final_status: str) -> str:
    if final_status == "explicit_positive":
        return "final_agreement" if gold == 1 else "final_positive_vs_gold_0"
    if final_status == "explicit_negative":
        return "final_agreement" if gold == 0 else "final_negative_vs_gold_1"
    return f"{final_status}_gold_{gold}"


def read_and_reprocess(source_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_uids: set[str] = set()

    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source_record = json.loads(line)
            uid = str(source_record["StudyInstanceUID"])
            if uid in seen_uids:
                raise ValueError(f"Duplicate StudyInstanceUID at line {line_number}: {uid}")
            seen_uids.add(uid)

            record = dict(source_record)
            record["source_prompt_version"] = source_record.get("prompt_version", "")
            record["policy_version"] = POLICY_VERSION
            record["postprocessed_targets"] = apply_v5_policy(source_record["extraction"])
            records.append(record)

    records.sort(key=lambda item: item["source_index"])
    if len(records) != 58:
        raise ValueError(f"Expected 58 v4 records, found {len(records)}")
    return records


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_tabular_outputs(
    records: list[dict[str, Any]], long_path: Path, summary_path: Path
) -> None:
    rows: list[dict[str, Any]] = []

    for record in records:
        for target in TARGETS:
            processed = record["postprocessed_targets"][target]
            gold = int(record["gold_labels"][target])
            rows.append(
                {
                    "source_index": record["source_index"],
                    "StudyInstanceUID": record["StudyInstanceUID"],
                    "report_language": record["extraction"]["report_language"],
                    "target": target,
                    "gold_label": gold,
                    "llm_status": processed["direct_status"],
                    "llm_binary": processed["direct_binary"],
                    "final_status": processed["final_status"],
                    "final_binary": processed["final_binary"],
                    "comparison": comparison_category(gold, processed["final_status"]),
                    "inference_source": processed["inference_source"],
                    "inference_rule": processed["inference_rule"],
                    "evidence": processed["direct_evidence"],
                    "related_evidence": "",
                    "source_prompt_version": record.get("source_prompt_version", ""),
                    "policy_version": POLICY_VERSION,
                    "model": record["model"],
                    "elapsed_seconds": record["generation"]["elapsed_seconds"],
                }
            )

    long_df = pd.DataFrame(rows).sort_values(["source_index", "target"])
    long_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    summary = (
        long_df.groupby("target", sort=False)
        .agg(
            reports=("StudyInstanceUID", "count"),
            gold_positive=("gold_label", "sum"),
            explicit_positive=("llm_status", lambda s: int((s == "explicit_positive").sum())),
            explicit_negative=("llm_status", lambda s: int((s == "explicit_negative").sum())),
            indeterminate=("llm_status", lambda s: int((s == "indeterminate").sum())),
            direct_not_mentioned=("llm_status", lambda s: int((s == "not_mentioned").sum())),
            final_positive=("final_status", lambda s: int((s == "explicit_positive").sum())),
            final_negative=("final_status", lambda s: int((s == "explicit_negative").sum())),
            final_not_mentioned=("final_status", lambda s: int((s == "not_mentioned").sum())),
            final_positive_vs_gold_0=(
                "comparison",
                lambda s: int((s == "final_positive_vs_gold_0").sum()),
            ),
            final_negative_vs_gold_1=(
                "comparison",
                lambda s: int((s == "final_negative_vs_gold_1").sum()),
            ),
        )
        .reset_index()
    )
    summary.insert(1, "policy_version", POLICY_VERSION)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")


def validate_outputs(records: list[dict[str, Any]]) -> None:
    for record in records:
        for target, processed in record["postprocessed_targets"].items():
            if processed["final_status"] != processed["direct_status"]:
                raise AssertionError(f"V5 changed a direct status: {target}")
            if processed["final_binary"] != processed["direct_binary"]:
                raise AssertionError(f"V5 changed a direct binary label: {target}")
            if processed["related_evidence"]:
                raise AssertionError(f"V5 contains related evidence: {target}")
            if processed["inference_source"] not in {"", "direct"}:
                raise AssertionError(f"V5 contains related inference: {target}")


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_jsonl).resolve()
    jsonl_path = Path(args.jsonl).resolve()
    long_path = Path(args.long_csv).resolve()
    summary_path = Path(args.summary_csv).resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"V4 source JSONL not found: {source_path}")

    records = read_and_reprocess(source_path)
    validate_outputs(records)
    write_jsonl(records, jsonl_path)
    write_tabular_outputs(records, long_path, summary_path)

    print(f"V5 policy: {POLICY_VERSION}")
    print(f"Source v4 JSONL: {source_path}")
    print(f"V5 JSONL: {jsonl_path} ({len(records)} reports)")
    print(f"V5 LONG CSV: {long_path} ({len(records) * len(TARGETS)} rows)")
    print(f"V5 SUMMARY CSV: {summary_path}")


if __name__ == "__main__":
    main()
