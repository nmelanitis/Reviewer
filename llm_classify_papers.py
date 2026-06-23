"""
Classify target-paper usage directly with a cloud LLM.

This is an optional Step 3 path that skips regex generation. It sends one paper
at a time to the selected provider and writes a classified_papers.csv-compatible
file for the existing review-prompt pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_client import LLMConfig, generate_text, require_api_key


DEFAULT_INPUT_CSV = "CitePaper.csv"
DEFAULT_METADATA_JSON = "CitePaper.meta.json"
DEFAULT_FULLTEXT_DIR = "open-pdfs-text"
DEFAULT_OUTPUT_CSV = "classified_papers.csv"
VERDICTS = {"yes", "no", "dont_know"}
CONFIDENCES = {"high", "medium", "low"}


@dataclass
class LLMClassification:
    verdict: str
    reason: str
    evidence_quote: str
    confidence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify citing papers by target-paper usage with a cloud LLM.",
    )
    parser.add_argument(
        "--input-csv",
        default=DEFAULT_INPUT_CSV,
        help=f"Input CSV from Step 1. Default: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_JSON,
        help=f"Target metadata JSON from Step 1. Default: {DEFAULT_METADATA_JSON}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output classified CSV. Default: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--evidence-mode",
        choices=["abstract", "fulltext"],
        default="abstract",
        help="Evidence sent per paper. Default: abstract.",
    )
    parser.add_argument(
        "--full-text-dir",
        default=DEFAULT_FULLTEXT_DIR,
        help=f"Extracted text directory for --evidence-mode fulltext. Default: {DEFAULT_FULLTEXT_DIR}",
    )
    parser.add_argument(
        "--abstract-chars",
        default="all",
        help="Maximum abstract characters per paper, or 'all'. Default: all.",
    )
    parser.add_argument(
        "--full-text-chars",
        type=int,
        default=30000,
        help="Maximum extracted full-text characters per paper. Default: 30000.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=800,
        help="Maximum LLM response tokens per paper. Default: 800.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Classify only the first N rows. Useful for smoke tests.",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Optional directory where raw per-paper LLM responses are saved.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per paper when the LLM call or JSON parsing fails. Default: 3.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=int,
        default=30,
        help="Seconds to sleep between retries. Default: 30.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing rows in --output and classify only missing papers.",
    )
    return parser.parse_args()


def clean(value: Any) -> str:
    text = "" if value is None else str(value)
    if text.lower() == "nan":
        return ""
    return " ".join(text.split())


def parse_char_limit(value: str, *, minimum: int, flag_name: str) -> int | None:
    if str(value).lower() == "all":
        return None
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"{flag_name} must be an integer or 'all'.") from None
    if parsed < minimum:
        raise ValueError(f"{flag_name} must be at least {minimum}, or 'all'.")
    return parsed


def cap_text(text: str, limit: int | None) -> str:
    if limit is None or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def load_metadata(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_full_text_index(full_text_dir: str) -> dict[str, dict[str, str]]:
    manifest_path = Path(full_text_dir) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Full-text manifest not found: {manifest_path}")

    index: dict[str, dict[str, str]] = {}
    for item in load_rows(str(manifest_path)):
        if item.get("status") not in {"extracted", "skipped_exists"}:
            continue
        text_path = item.get("text_path", "")
        if not text_path:
            continue
        if item.get("openalex_id"):
            index[item["openalex_id"]] = item
        if item.get("doi"):
            index[item["doi"]] = item
    return index


def full_text_for_row(
    row: dict[str, str],
    full_text_index: dict[str, dict[str, str]],
    limit: int,
) -> str:
    item = full_text_index.get(row.get("openalex_id", "")) or full_text_index.get(
        row.get("doi", "")
    )
    if not item:
        return ""
    text_path = Path(item.get("text_path", ""))
    if not text_path.exists():
        return ""
    return cap_text(clean(text_path.read_text(encoding="utf-8", errors="replace")), limit)


def build_classification_prompt(
    *,
    target: dict[str, Any],
    row: dict[str, str],
    abstract: str,
    evidence_source: str,
    full_text: str,
) -> str:
    evidence_parts = [
        "You are classifying whether a citing paper actually uses a target paper.",
        "",
        "Return only valid JSON with exactly these keys:",
        '{"verdict": "yes|no|dont_know", "reason": "...", "evidence_quote": "...", "confidence": "high|medium|low"}',
        "",
        "Label rules:",
        "- yes: the citing paper clearly uses, applies, extends, benchmarks, reproduces, adapts, or centrally discusses the target paper/method.",
        "- no: the citing paper only mentions the target as background/related work, is a review/survey, or has no target-specific usage evidence.",
        "- dont_know: evidence is missing, ambiguous, or insufficient.",
        "- If you would say maybe, return dont_know.",
        "- evidence_quote must be a short quote copied from the supplied evidence, or an empty string if no useful quote exists.",
        "- confidence must be high, medium, or low.",
        "",
        "Target paper:",
        f"- Title: {clean(target.get('display_name'))}",
        f"- DOI: {clean(target.get('doi'))}",
        f"- OpenAlex ID: {clean(target.get('openalex_id'))}",
        "",
        "Citing paper:",
        f"- Title: {clean(row.get('display_name'))}",
        f"- DOI: {clean(row.get('doi'))}",
        f"- Type: {clean(row.get('type'))}",
        f"- Publication year: {clean(row.get('publication_year'))}",
        f"- Evidence source: {evidence_source}",
        "",
        "Abstract:",
        abstract or "missing",
    ]
    if full_text:
        evidence_parts.extend(["", "Extracted full text, truncated if needed:", full_text])
    return "\n".join(evidence_parts)


def extract_json_object(response_text: str) -> dict[str, Any]:
    json_blocks = re.findall(r"```(?:json)?\s*(.*?)```", response_text, re.S | re.I)
    candidates = json_blocks + [response_text]
    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, re.S)
            if not match:
                continue
            try:
                value = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
        if isinstance(value, dict):
            return value
    raise ValueError("LLM response did not contain a JSON object.")


def normalize_verdict(value: Any) -> str:
    verdict = clean(value).lower().replace("-", "_").replace(" ", "_")
    if verdict in {"maybe", "unknown", "uncertain", "cannot_determine"}:
        return "dont_know"
    if verdict not in VERDICTS:
        return "dont_know"
    return verdict


def normalize_confidence(value: Any, verdict: str) -> str:
    confidence = clean(value).lower()
    if confidence in CONFIDENCES:
        return confidence
    return "low" if verdict == "dont_know" else "medium"


def parse_classification(response_text: str) -> LLMClassification:
    data = extract_json_object(response_text)
    verdict = normalize_verdict(data.get("verdict"))
    return LLMClassification(
        verdict=verdict,
        reason=clean(data.get("reason")) or "No reason returned by LLM.",
        evidence_quote=clean(data.get("evidence_quote")),
        confidence=normalize_confidence(data.get("confidence"), verdict),
    )


def row_key(row: dict[str, str]) -> str:
    return clean(row.get("openalex_id")) or clean(row.get("doi")) or clean(row.get("display_name"))


def write_classified_rows(path: str, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns(rows))
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in writer.fieldnames} for row in rows)


def load_existing_classifications(path: str, enabled: bool) -> dict[str, dict[str, str]]:
    if not enabled or not Path(path).exists():
        return {}
    existing = load_rows(path)
    return {row_key(row): row for row in existing if row_key(row)}


def classify_row(
    *,
    row: dict[str, str],
    target: dict[str, Any],
    config: LLMConfig,
    evidence_mode: str,
    abstract_chars: int | None,
    full_text_index: dict[str, dict[str, str]],
    full_text_chars: int,
    raw_dir: Path | None,
    row_number: int,
    retries: int,
    retry_sleep_seconds: int,
) -> dict[str, str]:
    abstract = cap_text(clean(row.get("abstract")), abstract_chars)
    full_text = ""
    if evidence_mode == "fulltext":
        full_text = full_text_for_row(row, full_text_index, full_text_chars)

    if evidence_mode == "abstract":
        evidence_source = "abstract" if abstract else "missing"
    elif full_text:
        evidence_source = "fulltext"
    elif abstract:
        evidence_source = "abstract_fallback"
    else:
        evidence_source = "missing"

    if evidence_source == "missing":
        classification = LLMClassification(
            verdict="dont_know",
            reason="No abstract or extracted full text available.",
            evidence_quote="",
            confidence="low",
        )
    else:
        prompt = build_classification_prompt(
            target=target,
            row=row,
            abstract=abstract,
            evidence_source=evidence_source,
            full_text=full_text,
        )
        attempts = max(retries, 0) + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = generate_text(config, prompt)
                if raw_dir:
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    raw_path = raw_dir / f"{row_number:04d}.md"
                    raw_path.write_text(response, encoding="utf-8")
                classification = parse_classification(response)
                break
            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    raise
                print(
                    f"Retrying paper {row_number} after error on attempt {attempt}/{attempts}: {exc}",
                    flush=True,
                )
                time.sleep(retry_sleep_seconds)
        else:
            raise RuntimeError(f"LLM classification failed: {last_error}")

    output = dict(row)
    output.update(
        {
            "verdict": classification.verdict,
            "reason": classification.reason,
            "matched_patterns": "",
            "classifier_mode": f"llm_{evidence_mode}",
            "evidence_source": evidence_source,
            "evidence_quote": classification.evidence_quote,
            "confidence": classification.confidence,
            "llm_provider": config.provider,
            "llm_model": config.model,
        }
    )
    return output


def output_columns(rows: list[dict[str, str]]) -> list[str]:
    preferred = [
        "openalex_id",
        "display_name",
        "doi",
        "abstract",
        "verdict",
        "reason",
        "matched_patterns",
        "classifier_mode",
        "evidence_source",
        "evidence_quote",
        "confidence",
        "llm_provider",
        "llm_model",
    ]
    existing = set().union(*(row.keys() for row in rows)) if rows else set(preferred)
    return [column for column in preferred if column in existing]


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        print("Error: --limit must be at least 1.", file=sys.stderr)
        return 2
    if args.full_text_chars < 200:
        print("Error: --full-text-chars must be at least 200.", file=sys.stderr)
        return 2
    if args.retries < 0:
        print("Error: --retries must be zero or greater.", file=sys.stderr)
        return 2
    if args.retry_sleep_seconds < 0:
        print("Error: --retry-sleep-seconds must be zero or greater.", file=sys.stderr)
        return 2

    try:
        abstract_chars = parse_char_limit(
            args.abstract_chars,
            minimum=100,
            flag_name="--abstract-chars",
        )
        metadata = load_metadata(args.metadata)
        target = metadata.get("target") or {}
        if not clean(target.get("display_name")):
            raise ValueError(f"No target.display_name found in {args.metadata}")

        rows = load_rows(args.input_csv)
        if args.limit is not None:
            rows = rows[: args.limit]

        full_text_index: dict[str, dict[str, str]] = {}
        if args.evidence_mode == "fulltext":
            full_text_index = load_full_text_index(args.full_text_dir)

        config = LLMConfig(
            provider=args.provider,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )
        require_api_key(args.provider)
        raw_dir = Path(args.raw_dir) if args.raw_dir else None

        existing_by_key = load_existing_classifications(args.output, args.resume)
        classified = []
        for index, row in enumerate(rows, start=1):
            key = row_key(row)
            if key in existing_by_key:
                print(f"Skipping {index}/{len(rows)} from existing output: {clean(row.get('display_name'))[:90]}", flush=True)
                classified.append(existing_by_key[key])
                continue

            print(f"Classifying {index}/{len(rows)}: {clean(row.get('display_name'))[:90]}", flush=True)
            classified.append(
                classify_row(
                    row=row,
                    target=target,
                    config=config,
                    evidence_mode=args.evidence_mode,
                    abstract_chars=abstract_chars,
                    full_text_index=full_text_index,
                    full_text_chars=args.full_text_chars,
                    raw_dir=raw_dir,
                    row_number=index,
                    retries=args.retries,
                    retry_sleep_seconds=args.retry_sleep_seconds,
                )
            )
            write_classified_rows(args.output, classified)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    write_classified_rows(args.output, classified)

    counts: dict[str, int] = {}
    for row in classified:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print(f"Target: {clean(target.get('display_name'))}")
    print(f"Mode: llm_{args.evidence_mode}")
    print(f"Rows classified: {len(classified)}")
    print("Classification summary: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
