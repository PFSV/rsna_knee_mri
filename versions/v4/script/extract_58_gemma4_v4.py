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
PROMPT_VERSION = "v4.0-direct-negative-and-synovitis-effusion"

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

TARGET_POLICIES = {
    "ACL": "direct_report_only",
    "MCL": "direct_report_only",
    "Medial Meniscus": "direct_report_only",
    "Lateral Meniscus": "direct_report_only",
    "Medial OA": "direct_report_only",
    "Lateral OA": "no_related_supplement",
    "PF OA": "candidate_for_later_validation_no_v4_supplement",
    "Effusion": "direct_report_only",
    "Synovitis": "supplement_from_direct_effusion_if_unmentioned",
    "Baker's": "no_related_supplement",
    "Contusion": "no_related_supplement",
    "Fracture": "keep_not_mentioned_without_direct_evidence",
}

TARGET_DEFINITIONS = """
- ACL: anterior cruciate ligament injury. A directly asserted sprain, injury, partial tear,
  complete tear, or rupture is positive. A directly scoped normal, intact, preserved, or
  no-tear/no-injury statement is negative. Mucoid degeneration or nonspecific signal alone is
  indirect unless the report calls it an injury. A suspected/possible injury is uncertain.
- MCL: medial collateral ligament injury, using the same direct positive/negative rules as ACL.
- Medial Meniscus: medial meniscus tear. A directly scoped normal meniscus or no-tear statement is
  negative. Degeneration or intrasubstance signal without a surface-reaching tear is indirect.
- Lateral Meniscus: lateral meniscus tear, using the same rules as Medial Meniscus.
- Medial OA: osteoarthritis of the medial tibiofemoral compartment. Explicit OA/arthrosis is
  positive. A directly scoped normal/preserved medial-compartment cartilage or explicit absence of
  medial OA/chondrosis is negative. Isolated abnormal cartilage, osteophyte, or subchondral change
  without an OA diagnosis is indirect rather than explicit positive.
- Lateral OA: osteoarthritis of the lateral tibiofemoral compartment, using the same rules and
  requiring lateral-compartment scope.
- PF OA: patellofemoral osteoarthritis, using the same rules and requiring patellofemoral scope.
  V4 does not supplement an unmentioned PF OA label from other findings.
- Effusion: knee joint effusion or excess intra-articular fluid. Trace/small effusion directly
  asserted is positive. No effusion/no excess joint fluid is negative.
- Synovitis: inflammation of the synovial lining. Explicit synovitis or language such as
  'synovial hypertrophy indicative of synovitis' is positive. Explicit no synovitis or a directly
  scoped normal synovium is negative. Nonspecific synovial thickening alone is indirect.
- Baker's: Baker's cyst / popliteal cyst. Direct presence is positive; explicit absence is negative.
- Contusion: bone contusion / bone bruise only. Muscle or soft-tissue contusion does not count.
  Direct no bone contusion/no bone bruise is negative. Traumatic marrow edema without a direct
  contusion/bruise assertion is indirect.
- Fracture: fracture, including stress or insufficiency fracture when directly asserted. Explicit
  no fracture/no osseous fracture is negative. If fracture is not discussed, it remains unmentioned.
""".strip()

SYSTEM_PROMPT = f"""
You are a conservative radiology report information-extraction system. Reports may be multilingual.
Analyze only the supplied report text. Extract direct target evidence first. Related-finding
supplementation is performed later by deterministic code, not by you.

Target definitions:
{TARGET_DEFINITIONS}

For every target, assign exactly one compact status code:
- P: the report directly asserts the target condition or an unambiguous synonym.
- N: the report directly denies the target condition, or directly states that the relevant
  structure/compartment is normal, intact, preserved, or without the target abnormality.
- U: the target itself is directly mentioned, but the assertion is suspected, possible, probable,
  equivocal, conflicting, or otherwise indeterminate.
- M: neither a direct positive, direct negative, nor direct uncertain statement is present.

Critical negative rules:
1. A correctly scoped 'normal', 'intact', 'preserved', 'unremarkable', '없음', '정상', 'no',
   'without', or equivalent multilingual statement is direct NEGATIVE evidence, not unrelated
   evidence and not M.
2. Examples: 'ACL and MCL are intact' means ACL=N and MCL=N. 'No medial or lateral meniscal tear'
   means both meniscus targets=N. 'No fracture or bone contusion' means Fracture=N and Contusion=N.
   'No joint effusion' means Effusion=N. 'No synovitis' means Synovitis=N.
3. Apply normal/absence statements only to targets actually covered by their anatomical and
   grammatical scope. Do not convert a vague global phrase into negatives for unrelated targets.

Other rules:
4. Do not treat missing mention as negative.
5. Do not use related findings to infer a target in the LLM response. In particular, effusion does
   not itself establish synovitis here; deterministic post-processing will handle that relation.
6. Respect anatomy and compartment. Medial, lateral, and patellofemoral findings are not
   interchangeable.
7. Distinguish bone contusion from muscle/soft-tissue contusion.
8. If direct positive and negative statements conflict, use U.
9. P, N, and U require one short exact quotation copied from the report in its original language.
   M requires an empty e. Never use N merely because a finding is absent from the report.
10. r must always be an empty string. Related-finding extraction is handled outside the model.
11. If a report says 'muscle contusion' and also 'no bone contusion', assign Contusion=N and quote
    the direct bone-contusion denial.
12. When unsure about direct evidence or negation scope, use U only for a direct uncertain mention
    of the target; otherwise use M.
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
        description=(
            "Extract direct 12-target evidence and apply the v4 Synovitis<-Effusion policy "
            "to 58 labeled reports."
        )
    )
    parser.add_argument("--input", default=str(PROJECT_DIR / "train.csv"))
    parser.add_argument(
        "--jsonl", default=str(EXTRACTION_DIR / "gemma4_e4b_extract_58_v4.jsonl")
    )
    parser.add_argument(
        "--long-csv", default=str(EXTRACTION_DIR / "gemma4_e4b_extract_58_v4_long.csv")
    )
    parser.add_argument(
        "--summary-csv",
        default=str(EXTRACTION_DIR / "gemma4_e4b_extract_58_v4_summary.csv"),
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
            extraction = json.loads(api_result["message"]["content"])
            enforce_direct_evidence_output(extraction, report)
            validate_extraction(extraction, report)
            metadata = {
                "elapsed_seconds": round(elapsed, 3),
                "prompt_eval_count": api_result.get("prompt_eval_count"),
                "eval_count": api_result.get("eval_count"),
                "total_duration_ns": api_result.get("total_duration"),
            }
            return extraction, metadata
        except urllib.error.HTTPError as exc:
            try:
                response_body = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:
                response_body = ""
            detail = f"HTTP {exc.code} {exc.reason}"
            if response_body:
                detail += f": {response_body}"
            last_error = RuntimeError(detail)
            if attempt < attempts:
                retry_note = (
                    "\n\nThe Ollama server returned an internal error. Retry the same extraction "
                    "with valid compact JSON and exact evidence quotations."
                )
                time.sleep(2 * attempt)
        except (KeyError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < attempts:
                retry_note = (
                    "\n\nYour previous response failed validation: "
                    f"{exc}. Return a corrected extraction. Correctly scoped normal/none wording is N, "
                    "P/N/U require one exact report quote in e, M requires an empty e, and r must be "
                    "empty. Do not infer from related findings."
                )
                time.sleep(2 * attempt)
    raise RuntimeError(f"Ollama extraction failed after {attempts} attempts: {last_error}")


def enforce_direct_evidence_output(extraction: dict[str, Any], report: str) -> None:
    """Keep the model layer strictly auditable and based on direct report evidence."""
    for item in extraction.get("targets", {}).values():
        item["r"] = ""
        if item.get("s") == "M":
            item["e"] = ""
            continue

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
            raise ValueError(f"Related findings must be empty in the model layer: {target}")


def apply_v4_policy(extraction: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Apply one-way, auditable post-processing without overwriting direct LLM judgments."""
    processed: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        item = extraction["targets"][target]
        direct_code = item["s"]
        direct_status = STATUS_CODES[direct_code]
        direct_binary = 1 if direct_code == "P" else 0 if direct_code == "N" else None
        processed[target] = {
            "direct_status": direct_status,
            "direct_binary": direct_binary,
            "direct_evidence": item["e"].strip(),
            "final_status": direct_status,
            "final_binary": direct_binary,
            "inference_source": "direct" if direct_code != "M" else "",
            "inference_rule": TARGET_POLICIES[target],
            "related_evidence": "",
        }

    synovitis = processed["Synovitis"]
    effusion_item = extraction["targets"]["Effusion"]

    # Direct Synovitis P/N/U always takes precedence. Only an unmentioned Synovitis target
    # may be supplemented, and only from a directly evidenced Effusion P or N.
    if synovitis["direct_status"] == "not_mentioned":
        if effusion_item["s"] == "P":
            synovitis.update(
                {
                    "final_status": "related_positive",
                    "final_binary": 1,
                    "inference_source": "Effusion",
                    "related_evidence": effusion_item["e"].strip(),
                }
            )
        elif effusion_item["s"] == "N":
            synovitis.update(
                {
                    "final_status": "related_negative",
                    "final_binary": 0,
                    "inference_source": "Effusion",
                    "related_evidence": effusion_item["e"].strip(),
                }
            )

    return processed


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                # Rebuild post-processing when loading so the deterministic policy is always current.
                item["postprocessed_targets"] = apply_v4_policy(item["extraction"])
                completed[item["StudyInstanceUID"]] = item
    return completed


def comparison_category(gold: int, final_status: str) -> str:
    if final_status in {"explicit_positive", "related_positive"}:
        return "final_agreement" if gold == 1 else "final_positive_vs_gold_0"
    if final_status in {"explicit_negative", "related_negative"}:
        return "final_agreement" if gold == 0 else "final_negative_vs_gold_1"
    return f"{final_status}_gold_{gold}"


def write_tabular_outputs(records: list[dict[str, Any]], long_path: Path, summary_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        report = record["Report"]
        processed_targets = record.get("postprocessed_targets") or apply_v4_policy(
            record["extraction"]
        )
        for target in TARGETS:
            processed = processed_targets[target]
            gold = int(record["gold_labels"][target])
            direct_evidence = processed["direct_evidence"]
            related_evidence = processed["related_evidence"]
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
                    "evidence": direct_evidence,
                    "related_evidence": related_evidence,
                    "evidence_quote_valid": (
                        (not direct_evidence) or exact_quote_is_present(direct_evidence, report)
                    ),
                    "related_evidence_quote_valid": (
                        (not related_evidence) or exact_quote_is_present(related_evidence, report)
                    ),
                    "model": record["model"],
                    "prompt_version": record["prompt_version"],
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
            direct_not_mentioned=("llm_status", lambda s: int((s == "not_mentioned").sum())),
            related_positive=("final_status", lambda s: int((s == "related_positive").sum())),
            related_negative=("final_status", lambda s: int((s == "related_negative").sum())),
            final_not_mentioned=("final_status", lambda s: int((s == "not_mentioned").sum())),
            final_positive_vs_gold_0=(
                "comparison",
                lambda s: int((s == "final_positive_vs_gold_0").sum()),
            ),
            final_negative_vs_gold_1=(
                "comparison",
                lambda s: int((s == "final_negative_vs_gold_1").sum()),
            ),
            invalid_direct_evidence=("evidence_quote_valid", lambda s: int((~s).sum())),
            invalid_related_evidence=(
                "related_evidence_quote_valid",
                lambda s: int((~s).sum()),
            ),
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
        postprocessed_targets = apply_v4_policy(extraction)
        record = {
            "source_index": int(source_index),
            "StudyInstanceUID": uid,
            "Report": str(row["Report"]),
            "gold_labels": {target: int(row[target]) for target in TARGETS},
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generation": generation,
            "extraction": extraction,
            "postprocessed_targets": postprocessed_targets,
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
