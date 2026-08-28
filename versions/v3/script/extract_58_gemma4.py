from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
VERSION_DIR = SCRIPT_DIR.parent
PROJECT_DIR = VERSION_DIR.parent.parent
EXTRACTION_DIR = VERSION_DIR / "extraction"

MODEL = "gemma4:e4b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
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

TARGET_DEFINITIONS = """
- ACL: anterior cruciate ligament injury. A directly asserted sprain, injury, partial tear,
  complete tear, or rupture is positive. Mucoid degeneration or nonspecific signal alone is
  indirect unless the report calls it an injury. A suspected/possible injury is uncertain.
- MCL: medial collateral ligament injury, using the same rules as ACL.
- Medial Meniscus: medial meniscus tear. Degeneration or intrasubstance signal without a
  surface-reaching tear is indirect, not an explicit tear.
- Lateral Meniscus: lateral meniscus tear, using the same rules as Medial Meniscus.
- Medial OA: osteoarthritis of the medial tibiofemoral compartment. Explicit OA/arthrosis is
  positive. Isolated cartilage loss, osteophyte, or subchondral change without an OA diagnosis
  is indirect.
- Lateral OA: osteoarthritis of the lateral tibiofemoral compartment, using the same rules.
- PF OA: patellofemoral osteoarthritis, using the same rules and requiring PF localization.
- Effusion: knee joint effusion or excess intra-articular fluid. Trace/small effusion still counts
  as explicit positive when directly asserted.
- Synovitis: inflammation of the synovial lining. Explicit synovitis or language such as
  'synovial hypertrophy indicative of synovitis' is positive. Nonspecific synovial thickening alone
  is indirect.
- Baker's: Baker's cyst / popliteal cyst.
- Contusion: bone contusion / bone bruise only. Muscle or soft-tissue contusion does not count.
  Traumatic marrow edema without a direct contusion/bruise assertion is indirect.
- Fracture: fracture, including stress or insufficiency fracture when directly asserted.
""".strip()

SYSTEM_PROMPT = f"""
You are a conservative radiology report information-extraction system. Reports may be multilingual.
Analyze only the supplied report text. This run is explicit-evidence-only: do not infer a target from
related, associated, secondary, or indirect findings. Do not invent findings and do not infer that an
unmentioned condition is negative.

Target definitions:
{TARGET_DEFINITIONS}

For every target, assign exactly one compact status code:
- P: the report directly asserts the target condition or an unambiguous synonym.
- N: the report directly denies the target condition or states the relevant
  structure is normal/intact, with correct negation scope.
- U: the target itself is directly mentioned, but the assertion is suspected, possible, probable,
  equivocal, conflicting, or otherwise indeterminate. U is not a positive or negative judgment.
- M: the target itself is not directly mentioned. Use M even when only related or indirect findings
  are present.

Rules:
1. Do not treat missing mention as negative.
2. Do not use related findings to infer any target in this run. Examples: effusion does not establish
   synovitis; marrow edema does not establish contusion; cartilage loss does not establish OA; and a
   meniscal finding does not establish compartmental OA.
3. Respect anatomy and compartment. Medial, lateral, and patellofemoral findings are not interchangeable.
4. Distinguish bone contusion from muscle/soft-tissue contusion.
5. If direct positive and negative statements conflict, use U.
6. Use P or N only when you can copy a direct exact quote into e. Never use N merely because a
   finding is absent from the report. If there is no relevant text, use M and leave e empty.
7. For U, e must quote a direct but uncertain/conflicting assertion about the target itself. An
   indirect finding alone must be M, not U. For M, e must be empty.
8. e must be one short exact quotation copied from the report in its original language.
9. r must always be an empty string. Related-finding extraction is disabled in this run.
10. Critical Contusion rule: muscle/soft-tissue contusion is NOT this target. If a report says
   'muscle contusion' and also 'no bone contusion', assign Contusion=N and quote 'no bone contusion'.
11. A target such as Fracture that is never discussed is M, never N.
12. When unsure whether evidence is direct and unambiguous, do not guess: use U only for a direct
    uncertain mention of the target; otherwise use M.
13. Do not add medical advice or explanations outside the requested JSON.
""".strip()


TARGET_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "s": {"type": "string", "enum": list(STATUS_CODES)},
        "e": {"type": "string"},
        "r": {"type": "string", "enum": [""]},
    },
    "required": ["s", "e", "r"],
    "additionalProperties": False,
}

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "report_language": {"type": "string"},
        "targets": {
            "type": "object",
            "properties": {target: TARGET_RESULT_SCHEMA for target in TARGETS},
            "required": TARGETS,
            "additionalProperties": False,
        },
    },
    "required": ["report_language", "targets"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conservatively extract explicit 12-target evidence from 58 labeled reports."
    )
    parser.add_argument("--input", default=str(PROJECT_DIR / "train.csv"))
    parser.add_argument(
        "--jsonl", default=str(EXTRACTION_DIR / "gemma4_e4b_extract_58_v3.jsonl")
    )
    parser.add_argument(
        "--long-csv", default=str(EXTRACTION_DIR / "gemma4_e4b_extract_58_v3_long.csv")
    )
    parser.add_argument(
        "--summary-csv",
        default=str(EXTRACTION_DIR / "gemma4_e4b_extract_58_v3_summary.csv"),
    )
    parser.add_argument(
        "--indices",
        default="",
        help="Optional comma-separated pandas source indices for a sample run.",
    )
    return parser.parse_args()


def collapse_whitespace(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def exact_quote_is_present(quote: str, report: str) -> bool:
    normalized_quote = collapse_whitespace(quote)
    return bool(normalized_quote) and normalized_quote in collapse_whitespace(report)


def request_extraction(report: str, attempts: int = 3) -> tuple[dict[str, Any], dict[str, Any]]:
    user_prompt = f"Extract all 12 targets from this knee MRI report.\n\n<REPORT>\n{report}\n</REPORT>"
    last_error: Exception | None = None
    retry_note = ""
    for attempt in range(1, attempts + 1):
        try:
            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt + retry_note},
                ],
                "stream": False,
                "think": False,
                "format": OUTPUT_SCHEMA,
                "options": {
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": 16384,
                    "num_predict": 2048,
                },
                "keep_alive": "30m",
            }
            request = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=900) as response:
                api_result = json.loads(response.read().decode("utf-8"))
            elapsed = time.perf_counter() - started
            content = api_result["message"]["content"]
            extraction = json.loads(content)
            enforce_explicit_only_output(extraction, report)
            validate_extraction(extraction, report)
            metadata = {
                "elapsed_seconds": round(elapsed, 3),
                "prompt_eval_count": api_result.get("prompt_eval_count"),
                "eval_count": api_result.get("eval_count"),
                "total_duration_ns": api_result.get("total_duration"),
            }
            return extraction, metadata
        except (KeyError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < attempts:
                retry_note = (
                    "\n\nYour previous response failed validation: "
                    f"{exc}. Return a corrected extraction. P/N/U require one exact report quote in e; "
                    "M requires an empty e, and r must always be empty. Do not infer from related findings."
                )
                time.sleep(2 * attempt)
    raise RuntimeError(f"Ollama extraction failed after {attempts} attempts: {last_error}")


def enforce_explicit_only_output(extraction: dict[str, Any], report: str) -> None:
    """Remove fields that must not carry an inferred judgment in explicit-only mode."""
    for item in extraction.get("targets", {}).values():
        # Related-finding extraction is disabled regardless of what the model emitted.
        item["r"] = ""
        # M means the target itself was not directly mentioned. An indirect finding may have
        # prompted the model to populate evidence, but retaining it would imply a judgment.
        if item.get("s") == "M":
            item["e"] = ""
            continue

        # A P/N/U status without a verbatim report quote is not auditable. Conservatively
        # downgrade it to M instead of guessing, retrying, or failing the entire report.
        evidence = str(item.get("e", "")).strip()
        if not evidence or not exact_quote_is_present(evidence, report):
            item["s"] = "M"
            item["e"] = ""


def validate_extraction(extraction: dict[str, Any], report: str) -> None:
    if set(extraction.get("targets", {})) != set(TARGETS):
        raise ValueError("The response does not contain exactly the 12 required targets.")
    for target in TARGETS:
        item = extraction["targets"][target]
        code = item.get("s")
        evidence = str(item.get("e", "")).strip()
        if code not in STATUS_CODES:
            raise ValueError(f"Invalid status code for {target}: {code}")
        if code == "M" and evidence:
            raise ValueError(f"Not-mentioned target has evidence for {target}: {evidence}")
        if code != "M" and not evidence:
            raise ValueError(f"Status {code} requires evidence for {target}")
        if evidence and not exact_quote_is_present(evidence, report):
            raise ValueError(f"Evidence is not an exact report quote for {target}: {evidence}")
        if str(item.get("r", "")).strip():
            raise ValueError(f"Related findings are disabled for this run: {target}")


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                completed[item["StudyInstanceUID"]] = item
    return completed


def comparison_category(gold: int, status: str) -> str:
    if status == "explicit_positive":
        return "direct_agreement" if gold == 1 else "explicit_positive_vs_gold_0"
    if status == "explicit_negative":
        return "direct_agreement" if gold == 0 else "explicit_negative_vs_gold_1"
    return f"{status}_gold_{gold}"


def write_tabular_outputs(records: list[dict[str, Any]], long_path: Path, summary_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        report = record["Report"]
        for target in TARGETS:
            item = record["extraction"]["targets"][target]
            status = STATUS_CODES[item["s"]]
            predicted = 1 if item["s"] == "P" else 0 if item["s"] == "N" else None
            gold = int(record["gold_labels"][target])
            evidence = item["e"].strip()
            evidence_valid = (not evidence) or exact_quote_is_present(evidence, report)
            rows.append(
                {
                    "source_index": record["source_index"],
                    "StudyInstanceUID": record["StudyInstanceUID"],
                    "report_language": record["extraction"]["report_language"],
                    "target": target,
                    "gold_label": gold,
                    "llm_status": status,
                    "llm_binary": predicted,
                    "comparison": comparison_category(gold, status),
                    "evidence": evidence,
                    "evidence_quote_valid": evidence_valid,
                    "related_findings": item["r"].strip(),
                    "model": record["model"],
                    "elapsed_seconds": record["generation"]["elapsed_seconds"],
                }
            )

    long_df = pd.DataFrame(rows).sort_values(["source_index", "target"])
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    summary = (
        long_df.groupby("target", sort=False)
        .agg(
            reports=("StudyInstanceUID", "count"),
            gold_positive=("gold_label", "sum"),
            explicit_positive=("llm_status", lambda s: int((s == "explicit_positive").sum())),
            explicit_negative=("llm_status", lambda s: int((s == "explicit_negative").sum())),
            indeterminate=("llm_status", lambda s: int((s == "indeterminate").sum())),
            not_mentioned=("llm_status", lambda s: int((s == "not_mentioned").sum())),
            explicit_positive_vs_gold_0=(
                "comparison",
                lambda s: int((s == "explicit_positive_vs_gold_0").sum()),
            ),
            explicit_negative_vs_gold_1=(
                "comparison",
                lambda s: int((s == "explicit_negative_vs_gold_1").sum()),
            ),
            invalid_evidence_rows=("evidence_quote_valid", lambda s: int((~s).sum())),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    jsonl_path = Path(args.jsonl).resolve()
    long_path = Path(args.long_csv).resolve()
    summary_path = Path(args.summary_csv).resolve()

    df = pd.read_csv(input_path, dtype={"StudyInstanceUID": "string", "Report": "string"})
    missing_targets = set(TARGETS) - set(df.columns)
    if missing_targets:
        raise ValueError(f"Missing target columns: {sorted(missing_targets)}")
    labeled = df.loc[df[TARGETS].notna().all(axis=1)].copy()
    if len(labeled) != 58:
        raise ValueError(f"Expected 58 fully labeled reports, found {len(labeled)}")

    if args.indices:
        selected_indices = {int(value.strip()) for value in args.indices.split(",") if value.strip()}
        labeled = labeled.loc[labeled.index.isin(selected_indices)]
        missing_indices = selected_indices - set(labeled.index)
        if missing_indices:
            raise ValueError(f"Requested indices are not fully labeled rows: {sorted(missing_indices)}")

    completed = load_completed(jsonl_path)
    total = len(labeled)
    for position, (source_index, row) in enumerate(labeled.iterrows(), start=1):
        uid = str(row["StudyInstanceUID"])
        if uid in completed:
            print(f"[{position}/{total}] skip source_index={source_index} (already extracted)", flush=True)
            continue

        print(f"[{position}/{total}] extract source_index={source_index}", flush=True)
        extraction, generation = request_extraction(str(row["Report"]))
        record = {
            "source_index": int(source_index),
            "StudyInstanceUID": uid,
            "Report": str(row["Report"]),
            "gold_labels": {target: int(row[target]) for target in TARGETS},
            "model": MODEL,
            "prompt_version": "v3.0-explicit-only",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generation": generation,
            "extraction": extraction,
        }
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        completed[uid] = record

    records = sorted(completed.values(), key=lambda item: item["source_index"])
    write_tabular_outputs(records, long_path, summary_path)
    print(f"JSONL: {jsonl_path} ({len(records)} reports)")
    print(f"LONG CSV: {long_path} ({len(records) * len(TARGETS)} rows)")
    print(f"SUMMARY CSV: {summary_path}")


if __name__ == "__main__":
    main()
