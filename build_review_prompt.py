"""
Build an offline LLM prompt for a target-paper usage review.

By default, the prompt includes only papers classified as "yes" because the
review is about how the target paper/model was used in the literature. Optional
flags can include uncertain or no-use examples. The script never calls an LLM or
the network; cloud LLM scripts consume the Markdown it writes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CLASSIFIED_CSV = "classified_papers.csv"
DEFAULT_METADATA_JSON = "CitePaper.meta.json"
DEFAULT_OUTPUT_MD = "review_prompt.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cautious abstract-only review-writing prompt.",
    )
    parser.add_argument(
        "--classified",
        default=DEFAULT_CLASSIFIED_CSV,
        help=f"Classified CSV from Step 3. Default: {DEFAULT_CLASSIFIED_CSV}",
    )
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_JSON,
        help=f"Target metadata JSON from Step 1. Default: {DEFAULT_METADATA_JSON}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_MD,
        help=f"Prompt Markdown output. Default: {DEFAULT_OUTPUT_MD}",
    )
    parser.add_argument(
        "--max-yes",
        type=int,
        default=100,
        help="Maximum yes papers to include. Default: 100.",
    )
    parser.add_argument(
        "--max-dont-know",
        type=int,
        default=20,
        help="Maximum dont_know papers to include. Default: 20.",
    )
    parser.add_argument(
        "--max-no",
        type=int,
        default=10,
        help="Maximum no papers to include when --include-no is set. Default: 10.",
    )
    parser.add_argument(
        "--include-dont-know",
        action="store_true",
        help="Include capped dont_know papers. Default: omit them.",
    )
    parser.add_argument(
        "--include-no",
        action="store_true",
        help="Include a small sample of no papers in the prompt.",
    )
    parser.add_argument(
        "--abstract-chars",
        default="900",
        help="Maximum abstract characters per paper, or 'all'. Default: 900.",
    )
    parser.add_argument(
        "--full-text-dir",
        default=None,
        help="Directory from extract_pdf_text.py, e.g. open-pdfs-text.",
    )
    parser.add_argument(
        "--full-text-chars",
        default="2000",
        help="Maximum extracted full-text characters per paper, or 'all'. Default: 2000.",
    )
    return parser.parse_args()


def load_metadata(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clean(value: str | None) -> str:
    return " ".join((value or "").split())


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


def abstract_snippet(row: dict[str, str], limit: int | None) -> str:
    abstract = clean(row.get("abstract"))
    if not abstract:
        return "missing"
    return cap_text(abstract, limit)


def load_full_text_index(full_text_dir: str | None) -> dict[str, dict[str, str]]:
    if not full_text_dir:
        return {}

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


def full_text_snippet(
    row: dict[str, str],
    full_text_index: dict[str, dict[str, str]],
    limit: int | None,
) -> str:
    item = full_text_index.get(row.get("openalex_id", "")) or full_text_index.get(
        row.get("doi", "")
    )
    if not item:
        return ""

    text_path = Path(item.get("text_path", ""))
    if not text_path.exists():
        return ""
    text = clean(text_path.read_text(encoding="utf-8", errors="replace"))
    return cap_text(text, limit)


def full_text_record_count(full_text_index: dict[str, dict[str, str]]) -> int:
    return len({item.get("text_path", "") for item in full_text_index.values() if item.get("text_path")})


def format_paper(
    row: dict[str, str],
    index: int,
    abstract_chars: int | None,
    full_text_index: dict[str, dict[str, str]],
    full_text_chars: int | None,
) -> str:
    title = clean(row.get("display_name"))
    doi = clean(row.get("doi")) or "missing"
    verdict = clean(row.get("verdict"))
    reason = clean(row.get("reason"))
    matched = clean(row.get("matched_patterns")) or "none"
    evidence_source = clean(row.get("evidence_source"))
    evidence_quote = clean(row.get("evidence_quote"))
    confidence = clean(row.get("confidence"))
    snippet = abstract_snippet(row, abstract_chars)
    full_text = full_text_snippet(row, full_text_index, full_text_chars)
    lines = [
        f"{index}. **{title}**",
        f"   - DOI: {doi}",
        f"   - Verdict: {verdict}",
        f"   - Classifier reason: {reason}",
        f"   - Matched patterns: {matched}",
    ]
    if evidence_source:
        lines.append(f"   - LLM evidence source: {evidence_source}")
    if confidence:
        lines.append(f"   - LLM confidence: {confidence}")
    if evidence_quote:
        lines.append(f"   - LLM evidence quote: {evidence_quote}")
    lines.extend(
        [
            f"   - Abstract evidence: {snippet}",
            f"   - Full-text evidence: {full_text or 'not available in extracted text set'}",
        ]
    )
    return "\n".join(lines)


def section_for_rows(
    title: str,
    rows: list[dict[str, str]],
    *,
    limit: int,
    abstract_chars: int | None,
    full_text_index: dict[str, dict[str, str]],
    full_text_chars: int | None,
) -> list[str]:
    selected = rows[:limit]
    lines = [f"## {title}", ""]
    if not selected:
        lines.append("No papers included in this section.")
        lines.append("")
        return lines

    lines.append(
        "\n\n".join(
            format_paper(
                row,
                index + 1,
                abstract_chars,
                full_text_index,
                full_text_chars,
            )
            for index, row in enumerate(selected)
        )
    )
    lines.append("")
    if len(rows) > limit:
        lines.append(f"Note: {len(rows) - limit} additional papers in this group were omitted by limit.")
        lines.append("")
    return lines


def render_prompt(
    *,
    metadata: dict[str, Any],
    rows: list[dict[str, str]],
    include_dont_know: bool,
    include_no: bool,
    max_yes: int,
    max_dont_know: int,
    max_no: int,
    abstract_chars: int | None,
    full_text_index: dict[str, dict[str, str]],
    full_text_chars: int | None,
) -> str:
    target = metadata.get("target") or {}
    verdict_counts = Counter(row.get("verdict", "missing") for row in rows)
    mode_counts = Counter(row.get("classifier_mode", "missing") for row in rows)
    yes_rows = [row for row in rows if row.get("verdict") == "yes"]
    dont_know_rows = [row for row in rows if row.get("verdict") == "dont_know"]
    no_rows = [row for row in rows if row.get("verdict") == "no"]

    lines = [
        "# Review-Writing Prompt",
        "",
        "You are writing an evidence-based review section about how later papers use a target paper.",
        "Use only the evidence below. By default it lists papers classified as confirmed usage candidates. It is based on metadata, abstracts, classifier audit fields, and any extracted full-text snippets included here.",
        "Use cautious language. Separate confirmed usage from uncertain usage. Do not claim a paper uses the target method unless the provided abstract evidence supports it.",
        "",
        "## Target Paper",
        f"- Title: {target.get('display_name', '')}",
        f"- DOI: {target.get('doi', '')}",
        f"- OpenAlex ID: {target.get('openalex_id', '')}",
        f"- Publication year: {target.get('publication_year', '')}",
        f"- OpenAlex cited_by_count at fetch time: {target.get('cited_by_count', '')}",
        "",
        "## Classification Summary",
        f"- Total classified papers: {len(rows)}",
        "- Verdict counts: "
        + ", ".join(f"{key}={value}" for key, value in verdict_counts.most_common()),
        "- Classifier modes: "
        + ", ".join(f"{key}={value}" for key, value in mode_counts.most_common()),
        f"- Extracted full-text records available in prompt: {full_text_record_count(full_text_index)}",
        "",
        "## Writing Task",
        "Write a review-style section with these parts:",
        "1. A short overview of the citation landscape.",
        "2. A synthesis of confirmed uses/extensions/applications of the target paper.",
        "3. A limitations paragraph stating whether the draft is based on abstracts, extracted full text, or both.",
        "4. A bullet list of papers that should be manually checked next.",
        "",
    ]
    lines.extend(
        section_for_rows(
            "Confirmed Usage Candidates",
            yes_rows,
            limit=max_yes,
            abstract_chars=abstract_chars,
            full_text_index=full_text_index,
            full_text_chars=full_text_chars,
        )
    )
    if include_dont_know:
        lines.extend(
            section_for_rows(
                "Uncertain Usage Candidates",
                dont_know_rows,
                limit=max_dont_know,
                abstract_chars=abstract_chars,
                full_text_index=full_text_index,
                full_text_chars=full_text_chars,
            )
        )
    else:
        lines.extend(
            [
                "## Uncertain Usage Candidates",
                "",
                f"{len(dont_know_rows)} papers were classified as dont_know and are not listed by default.",
                "Use `--include-dont-know` when generating the prompt if you want a capped sample included.",
                "",
            ]
        )
    if include_no:
        lines.extend(
            section_for_rows(
                "Likely Background-Only / No Usage Sample",
                no_rows,
                limit=max_no,
                abstract_chars=abstract_chars,
                full_text_index=full_text_index,
                full_text_chars=full_text_chars,
            )
        )
    else:
        lines.extend(
            [
                "## Likely Background-Only / No Usage Papers",
                "",
                f"{len(no_rows)} papers were classified as no and are not listed by default.",
                "Use `--include-no` when generating the prompt if you want a sample included.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.max_yes < 0 or args.max_dont_know < 0 or args.max_no < 0:
        print("Limits must be zero or greater.")
        return 2
    try:
        abstract_chars = parse_char_limit(args.abstract_chars, minimum=100, flag_name="--abstract-chars")
        full_text_chars = parse_char_limit(args.full_text_chars, minimum=200, flag_name="--full-text-chars")
    except ValueError as exc:
        print(exc)
        return 2

    metadata = load_metadata(args.metadata)
    rows = load_rows(args.classified)
    full_text_index = load_full_text_index(args.full_text_dir)
    prompt = render_prompt(
        metadata=metadata,
        rows=rows,
        include_dont_know=args.include_dont_know,
        include_no=args.include_no,
        max_yes=args.max_yes,
        max_dont_know=args.max_dont_know,
        max_no=args.max_no,
        abstract_chars=abstract_chars,
        full_text_index=full_text_index,
        full_text_chars=full_text_chars,
    )
    Path(args.output).write_text(prompt, encoding="utf-8")

    counts = Counter(row.get("verdict", "missing") for row in rows)
    print(f"Rows read: {len(rows)}")
    print("Verdict counts: " + ", ".join(f"{key}={value}" for key, value in counts.most_common()))
    print(f"Full-text records available: {full_text_record_count(full_text_index)}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
