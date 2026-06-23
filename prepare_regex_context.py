"""
Prepare offline context for building paper-usage regexes.

Step 2 of the project creates a compact, pasteable context file for a GPT or
local LLM. It does not classify papers permanently; it gathers evidence that
helps a human/model design regex rules for whether citing papers actually use
the target paper from CitePaper.meta.json.

Examples:
    python3 prepare_regex_context.py
    python3 prepare_regex_context.py --alias MAE --alias "masked autoencoder"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT_CSV = "CitePaper.csv"
DEFAULT_METADATA_JSON = "CitePaper.meta.json"
DEFAULT_OUTPUT_MD = "regex_context.md"
DEFAULT_MAX_EXAMPLES = 8
DEFAULT_SNIPPET_CHARS = 700

STOPWORDS = {
    "about",
    "after",
    "against",
    "also",
    "among",
    "are",
    "based",
    "because",
    "been",
    "being",
    "between",
    "both",
    "could",
    "does",
    "from",
    "have",
    "into",
    "more",
    "most",
    "only",
    "other",
    "over",
    "paper",
    "papers",
    "scalable",
    "show",
    "shows",
    "such",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "through",
    "using",
    "were",
    "when",
    "where",
    "which",
    "while",
    "with",
    "within",
    "would",
}

USAGE_CUES = [
    "adopt",
    "adapt",
    "apply",
    "baseline",
    "benchmark",
    "build",
    "compare",
    "extend",
    "fine-tune",
    "finetune",
    "inspired",
    "pretrain",
    "pre-train",
    "reproduce",
    "use",
    "utilize",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a compact context file for GPT/local regex generation.",
    )
    parser.add_argument(
        "--input-csv",
        default=DEFAULT_INPUT_CSV,
        help=f"Citing-paper CSV from Step 1. Default: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_JSON,
        help=f"Step 1 metadata JSON. Default: {DEFAULT_METADATA_JSON}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_MD,
        help=f"Markdown context output. Default: {DEFAULT_OUTPUT_MD}",
    )
    parser.add_argument(
        "--target-title",
        default=None,
        help="Override the target paper title from metadata.",
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        help="Target-paper alias/keyword. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=DEFAULT_MAX_EXAMPLES,
        help="Maximum examples per evidence bucket.",
    )
    parser.add_argument(
        "--snippet-chars",
        type=int,
        default=DEFAULT_SNIPPET_CHARS,
        help="Maximum abstract characters per example.",
    )
    return parser.parse_args()


def load_metadata(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def tokenize(value: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", value.lower())


def title_terms(title: str) -> list[str]:
    terms = []
    for token in tokenize(title):
        if token not in STOPWORDS and len(token) >= 4:
            terms.append(token)
            if token.endswith("s") and len(token) > 5:
                terms.append(token[:-1])
    return sorted(set(terms))


def title_phrases(title: str) -> list[str]:
    words = [word for word in tokenize(title) if word not in STOPWORDS]
    phrases = set()
    for size in (2, 3):
        for index in range(0, max(len(words) - size + 1, 0)):
            phrase = " ".join(words[index : index + size])
            if len(phrase) >= 8:
                phrases.add(phrase)
    return sorted(phrases)


def text_for_row(row: dict[str, str]) -> str:
    return normalize_space(f"{row.get('display_name', '')} {row.get('abstract', '')}")


def compile_patterns(values: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    patterns = []
    for value in values:
        cleaned = normalize_space(value)
        if not cleaned:
            continue
        escaped = re.escape(cleaned).replace(r"\ ", r"\s+")
        patterns.append((cleaned, re.compile(rf"\b{escaped}\b", re.I)))
    return patterns


def score_row(
    row: dict[str, str],
    *,
    full_title: str,
    aliases: list[str],
    terms: list[str],
    phrases: list[str],
) -> tuple[int, list[str]]:
    text = text_for_row(row)
    score = 0
    reasons: list[str] = []

    if re.search(re.escape(full_title), text, re.I):
        score += 6
        reasons.append("full target title")

    for alias, pattern in compile_patterns(aliases):
        if pattern.search(text):
            score += 5
            reasons.append(f"alias: {alias}")

    for phrase, pattern in compile_patterns(phrases):
        if pattern.search(text):
            score += 3
            reasons.append(f"title phrase: {phrase}")

    matched_terms = [term for term in terms if re.search(rf"\b{re.escape(term)}\b", text, re.I)]
    if matched_terms:
        score += min(len(set(matched_terms)), 4)
        reasons.append("title terms: " + ", ".join(sorted(set(matched_terms))[:6]))

    if matched_terms and any(re.search(rf"\b{cue}\w*\b", text, re.I) for cue in USAGE_CUES):
        score += 2
        reasons.append("usage cue near target vocabulary")

    return score, reasons


def categorize_rows(
    rows: list[dict[str, str]],
    *,
    full_title: str,
    aliases: list[str],
    terms: list[str],
    phrases: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scored = []
    for row in rows:
        score, reasons = score_row(
            row,
            full_title=full_title,
            aliases=aliases,
            terms=terms,
            phrases=phrases,
        )
        scored.append({"row": row, "score": score, "reasons": reasons})

    likely_positive = [
        item for item in scored if item["score"] >= 5 and item["row"].get("abstract")
    ]
    likely_uncertain = [
        item
        for item in scored
        if (item["score"] > 0 and item["score"] < 5) or not item["row"].get("abstract")
    ]
    likely_negative = [item for item in scored if item["score"] == 0 and item["row"].get("abstract")]

    likely_positive.sort(key=lambda item: item["score"], reverse=True)
    likely_uncertain.sort(key=lambda item: item["score"], reverse=True)
    likely_negative.sort(key=lambda item: item["row"].get("cited_by_count", ""), reverse=True)
    return likely_positive, likely_uncertain, likely_negative


def common_terms(rows: list[dict[str, str]], target_terms_: list[str]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(
            token
            for token in tokenize(text_for_row(row))
            if token not in STOPWORDS and len(token) >= 4
        )
    ignore = set(target_terms_)
    return [(term, count) for term, count in counter.most_common(40) if term not in ignore][:25]


def format_example(item: dict[str, Any], index: int, snippet_chars: int) -> str:
    row = item["row"]
    title = normalize_space(row.get("display_name", ""))
    abstract = normalize_space(row.get("abstract", ""))
    snippet = abstract[:snippet_chars]
    if len(abstract) > snippet_chars:
        snippet += "..."
    doi = row.get("doi", "")
    reasons = "; ".join(item["reasons"]) if item["reasons"] else "no target-specific terms"

    lines = [
        f"{index}. **{title}**",
        f"   - DOI: {doi or 'missing'}",
        f"   - Heuristic score: {item['score']} ({reasons})",
    ]
    if snippet:
        lines.append(f"   - Abstract snippet: {snippet}")
    else:
        lines.append("   - Abstract snippet: missing")
    return "\n".join(lines)


def render_context(
    *,
    metadata: dict[str, Any],
    rows: list[dict[str, str]],
    target_title: str,
    aliases: list[str],
    terms: list[str],
    phrases: list[str],
    output_path: str,
    max_examples: int,
    snippet_chars: int,
) -> str:
    positives, uncertain, negatives = categorize_rows(
        rows,
        full_title=target_title,
        aliases=aliases,
        terms=terms,
        phrases=phrases,
    )
    target = metadata.get("target", {})
    abstract_count = sum(1 for row in rows if row.get("abstract"))
    type_counts = Counter(row.get("type", "") or "missing" for row in rows)
    year_counts = Counter(row.get("publication_year", "") or "missing" for row in rows)
    frequent_terms = common_terms(rows, terms)

    sections = [
        "# Paper-Usage Regex Context",
        "",
        "## Target Paper",
        f"- Title: {target_title}",
        f"- DOI: {target.get('doi', '')}",
        f"- OpenAlex ID: {target.get('openalex_id', '')}",
        f"- Fetched citation rows: {len(rows)}",
        f"- Rows with abstracts: {abstract_count}",
        f"- Output file: {output_path}",
        "",
        "## Classification Goal",
        "- yes: citing paper clearly uses, applies, extends, benchmarks, reproduces, adapts, or centrally discusses the target paper/method.",
        "- no: citing paper only cites the target in background/related work, or has no target-specific evidence in title/abstract.",
        "- dont_know: abstract/title is missing, ambiguous, or does not provide enough evidence about the citation role.",
        "",
        "## Target Vocabulary Seeds",
        f"- Aliases supplied by user: {', '.join(aliases) if aliases else 'none'}",
        f"- Title-derived phrases: {', '.join(phrases) if phrases else 'none'}",
        f"- Title-derived terms: {', '.join(terms) if terms else 'none'}",
        "",
        "## Citation Set Snapshot",
        "- Paper types: " + ", ".join(f"{key}={value}" for key, value in type_counts.most_common()),
        "- Publication years: " + ", ".join(f"{key}={value}" for key, value in year_counts.most_common()),
        "- Frequent non-target terms: "
        + ", ".join(f"{term} ({count})" for term, count in frequent_terms),
        "",
        "## Candidate Positive Examples",
        "",
        "\n\n".join(
            format_example(item, index + 1, snippet_chars)
            for index, item in enumerate(positives[:max_examples])
        )
        or "No likely positive examples found by simple heuristics.",
        "",
        "## Candidate Uncertain Examples",
        "",
        "\n\n".join(
            format_example(item, index + 1, snippet_chars)
            for index, item in enumerate(uncertain[:max_examples])
        )
        or "No uncertain examples found by simple heuristics.",
        "",
        "## Candidate Negative Examples",
        "",
        "\n\n".join(
            format_example(item, index + 1, snippet_chars)
            for index, item in enumerate(negatives[:max_examples])
        )
        or "No likely negative examples found by simple heuristics.",
        "",
        "## Requested LLM Output",
        "- Propose regex groups for yes/no/dont_know evidence.",
        "- Explain the intended precision/recall tradeoff.",
        "- Include classification decision priority.",
        "- Include examples of papers that each rule should catch.",
        "- Keep regexes Python-compatible for the `re` module.",
        "",
    ]
    return "\n".join(sections)


def main() -> int:
    args = parse_args()
    if args.max_examples < 1:
        print("--max-examples must be at least 1.")
        return 2
    if args.snippet_chars < 100:
        print("--snippet-chars must be at least 100.")
        return 2

    metadata = load_metadata(args.metadata)
    target = metadata.get("target", {})
    target_title = args.target_title or target.get("display_name") or ""
    if not target_title:
        print("Could not find target title. Pass --target-title or check metadata.")
        return 1

    rows = load_rows(args.input_csv)
    terms = title_terms(target_title)
    phrases = title_phrases(target_title)
    aliases = [normalize_space(alias) for alias in args.alias if normalize_space(alias)]

    context = render_context(
        metadata=metadata,
        rows=rows,
        target_title=target_title,
        aliases=aliases,
        terms=terms,
        phrases=phrases,
        output_path=args.output,
        max_examples=args.max_examples,
        snippet_chars=args.snippet_chars,
    )
    Path(args.output).write_text(context, encoding="utf-8")

    print(f"Target: {target_title}")
    print(f"Citing papers: {len(rows)}")
    print(f"Rows with abstracts: {sum(1 for row in rows if row.get('abstract'))}")
    print(f"Aliases: {', '.join(aliases) if aliases else 'none'}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
