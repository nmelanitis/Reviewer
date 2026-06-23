"""
Target-paper usage classifier.

Step 3 classifies papers from CitePaper.csv according to whether each citing
paper appears to use the target paper recorded in CitePaper.meta.json.

Default metadata mode:
    python3 keywords.py
    python3 keywords.py --alias MAE --alias "masked autoencoder"

Generated regex mode:
    python3 keywords.py --mode generated --regexp regexpfile.py

The generated regex file is trusted local Python. It may define:
    STRONG_YES = [...]
    WEAK_YES = [...]
    BACKGROUND_ONLY = [...]
    AMBIGUOUS = [...]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_INPUT_CSV = "CitePaper.csv"
DEFAULT_METADATA_JSON = "CitePaper.meta.json"
DEFAULT_OUTPUT_CSV = "classified_papers.csv"

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
    r"\badapt\w*\b",
    r"\badopt\w*\b",
    r"\bappl(?:y|ied|ies|ication)\b",
    r"\bbaseline\w*\b",
    r"\bbenchmark\w*\b",
    r"\bbuild(?:s|ing)?\b",
    r"\bbuilt\b",
    r"\bcompar(?:e|ed|es|ing|ison)\b",
    r"\bdeploy\w*\b",
    r"\bdevelop\w*\b",
    r"\bextend\w*\b",
    r"\bfine.?tun\w*\b",
    r"\binspir(?:e|ed|es|ing)\b",
    r"\bpre.?train\w*\b",
    r"\bpropos(?:e|ed|es|ing)\b",
    r"\breproduc\w*\b",
    r"\btrain\w*\b",
    r"\buse[sd]?\b",
    r"\butili[sz]\w*\b",
]

BACKGROUND_ONLY = [
    r"\brecent (work|studies|advances)\b",
    r"\bhas gained (significant|much|considerable)? ?attention\b",
    r"\bhave achieved (great|significant|remarkable|substantial)? ?success\b",
    r"\bis widely used\b",
    r"\bpopularized by\b",
    r"\bmotivated by recent\b",
]

REVIEW_MARKERS = [
    r"\bsystematic review\b",
    r"\bliterature review\b",
    r"\bscoping review\b",
    r"\bumbrella review\b",
    r"\bnarrative review\b",
    r"\bmeta.?analysis\b",
    r"\bsurvey\b",
]


@dataclass
class Classification:
    verdict: str
    reason: str
    matched_patterns: list[str]


@dataclass
class PatternSet:
    strong_yes: list[str]
    weak_yes: list[str]
    background_only: list[str]
    ambiguous: list[str]


@dataclass
class CompiledPatternSet:
    originals: PatternSet
    strong_yes: list[re.Pattern[str]]
    weak_yes: list[re.Pattern[str]]
    background_only: list[re.Pattern[str]]
    ambiguous: list[re.Pattern[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify citing papers by target-paper usage.",
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
    parser.add_argument(
        "--mode",
        choices=["metadata", "generated"],
        default="metadata",
        help="Classifier mode. Default: metadata",
    )
    parser.add_argument(
        "--regexp",
        default=None,
        help="Trusted Python regex file for --mode generated.",
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        help="Target-paper alias/keyword. Can be passed multiple times.",
    )
    return parser.parse_args()


def load_metadata(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def tokenize(value: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", value.lower())


def text_for_row(row: pd.Series) -> tuple[str, str, str]:
    title = normalize_space(row.get("display_name", ""))
    abstract = normalize_space(row.get("abstract", ""))
    if abstract.lower() == "nan":
        abstract = ""
    return title, abstract, normalize_space(f"{title} {abstract}")


def important_title_terms(title: str) -> list[str]:
    terms: set[str] = set()
    for token in tokenize(title):
        if token not in STOPWORDS and len(token) >= 4:
            terms.add(token)
            if token.endswith("s") and len(token) > 5:
                terms.add(token[:-1])
    return sorted(terms)


def title_phrases(title: str) -> list[str]:
    words = [word for word in tokenize(title) if word not in STOPWORDS]
    phrases: set[str] = set()
    for size in (2, 3):
        for index in range(0, max(len(words) - size + 1, 0)):
            phrase = " ".join(words[index : index + size])
            if len(phrase) >= 8:
                phrases.add(phrase)
    return sorted(phrases)


def phrase_regex(value: str) -> str:
    escaped = re.escape(normalize_space(value))
    return escaped.replace(r"\ ", r"\s+")


def compile_pattern_list(name: str, patterns: Iterable[str]) -> list[re.Pattern[str]]:
    compiled = []
    for pattern in patterns:
        if not isinstance(pattern, str):
            raise ValueError(f"{name} contains a non-string pattern: {pattern!r}")
        try:
            compiled.append(re.compile(pattern, re.I))
        except re.error as exc:
            raise ValueError(f"Invalid regex in {name}: {pattern!r} ({exc})") from exc
    return compiled


def matching_patterns(
    compiled: list[re.Pattern[str]],
    originals: list[str],
    text: str,
) -> list[str]:
    return [pattern for pattern, regex in zip(originals, compiled) if regex.search(text)]


def compile_pattern_set(patterns: PatternSet, prefix: str = "") -> CompiledPatternSet:
    label = f"{prefix} " if prefix else ""
    return CompiledPatternSet(
        originals=patterns,
        strong_yes=compile_pattern_list(f"{label}STRONG_YES", patterns.strong_yes),
        weak_yes=compile_pattern_list(f"{label}WEAK_YES", patterns.weak_yes),
        background_only=compile_pattern_list(
            f"{label}BACKGROUND_ONLY", patterns.background_only
        ),
        ambiguous=compile_pattern_list(f"{label}AMBIGUOUS", patterns.ambiguous),
    )


def load_generated_patterns(path: str | None) -> PatternSet:
    if not path:
        raise ValueError("--regexp is required when --mode generated")

    regexp_path = Path(path)
    if not regexp_path.exists():
        raise ValueError(f"Regex file does not exist: {path}")

    spec = importlib.util.spec_from_file_location("paper_usage_patterns", regexp_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load regex file: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    strong_yes = list(getattr(module, "STRONG_YES", []))
    weak_yes = list(getattr(module, "WEAK_YES", []))
    background_only = list(getattr(module, "BACKGROUND_ONLY", []))
    ambiguous = list(getattr(module, "AMBIGUOUS", []))

    if not strong_yes and not weak_yes:
        raise ValueError("Generated regex file must define STRONG_YES and/or WEAK_YES.")

    for name, values in [
        ("STRONG_YES", strong_yes),
        ("WEAK_YES", weak_yes),
        ("BACKGROUND_ONLY", background_only),
        ("AMBIGUOUS", ambiguous),
    ]:
        compile_pattern_list(name, values)

    return PatternSet(
        strong_yes=strong_yes,
        weak_yes=weak_yes,
        background_only=background_only,
        ambiguous=ambiguous,
    )


def generated_classify(row: pd.Series, patterns: CompiledPatternSet) -> Classification:
    title, abstract, full_text = text_for_row(row)
    paper_type = normalize_space(row.get("type", "")).lower()
    if not abstract:
        return Classification("dont_know", "No abstract available", [])

    originals = patterns.originals
    strong_matches = matching_patterns(
        patterns.strong_yes, originals.strong_yes, full_text
    )
    weak_matches = matching_patterns(patterns.weak_yes, originals.weak_yes, full_text)
    background_matches = matching_patterns(
        patterns.background_only, originals.background_only, full_text
    )
    ambiguous_matches = matching_patterns(
        patterns.ambiguous, originals.ambiguous, full_text
    )

    if strong_matches:
        return Classification("yes", "Strong generated regex evidence", strong_matches)

    if weak_matches and not background_matches:
        return Classification("yes", "Weak generated regex evidence", weak_matches)

    if paper_type == "review" or any(re.search(pattern, full_text, re.I) for pattern in REVIEW_MARKERS):
        matches = background_matches or ["paper type/review marker"]
        return Classification("no", "Review or survey/background paper", matches)

    if background_matches and not weak_matches:
        return Classification("no", "Background-only generated regex evidence", background_matches)

    if weak_matches or ambiguous_matches:
        return Classification(
            "dont_know",
            "Ambiguous generated regex evidence",
            weak_matches + ambiguous_matches + background_matches,
        )

    return Classification("no", "No generated target-usage evidence", [])


def metadata_patterns(target_title: str, aliases: list[str]) -> PatternSet:
    alias_patterns = [rf"\b{phrase_regex(alias)}\b" for alias in aliases if normalize_space(alias)]
    phrase_patterns = [rf"\b{phrase_regex(phrase)}\b" for phrase in title_phrases(target_title)]
    term_patterns = [rf"\b{re.escape(term)}\b" for term in important_title_terms(target_title)]

    strong_yes = alias_patterns + phrase_patterns
    weak_yes = term_patterns
    return PatternSet(
        strong_yes=strong_yes,
        weak_yes=weak_yes,
        background_only=BACKGROUND_ONLY,
        ambiguous=[],
    )


def metadata_classify(
    row: pd.Series,
    patterns: CompiledPatternSet,
    usage_patterns: list[re.Pattern[str]],
) -> Classification:
    title, abstract, full_text = text_for_row(row)
    paper_type = normalize_space(row.get("type", "")).lower()
    if not abstract:
        return Classification("dont_know", "No abstract available", [])

    originals = patterns.originals
    strong_matches = matching_patterns(
        patterns.strong_yes, originals.strong_yes, full_text
    )
    weak_matches = matching_patterns(patterns.weak_yes, originals.weak_yes, full_text)
    usage_matches = matching_patterns(usage_patterns, USAGE_CUES, full_text)
    background_matches = matching_patterns(
        patterns.background_only, originals.background_only, full_text
    )

    if strong_matches and usage_matches:
        return Classification(
            "yes",
            "Target alias/phrase plus usage cue",
            strong_matches + usage_matches[:3],
        )

    if strong_matches and not background_matches:
        return Classification(
            "yes",
            "Strong target alias/phrase evidence",
            strong_matches,
        )

    if weak_matches and usage_matches:
        return Classification(
            "dont_know",
            "Weak target terms plus usage cue",
            weak_matches[:5] + usage_matches[:3],
        )

    if paper_type == "review" or any(re.search(pattern, full_text, re.I) for pattern in REVIEW_MARKERS):
        matches = background_matches or ["paper type/review marker"]
        return Classification("no", "Review or survey/background paper", matches)

    if weak_matches:
        return Classification("dont_know", "Weak target vocabulary only", weak_matches[:5])

    return Classification("no", "No target-specific evidence", [])


def classify_dataframe(
    df: pd.DataFrame,
    *,
    mode: str,
    patterns: CompiledPatternSet,
) -> pd.DataFrame:
    usage_patterns = compile_pattern_list("usage cues", USAGE_CUES)
    classifications = []
    for _, row in df.iterrows():
        if mode == "generated":
            classifications.append(generated_classify(row, patterns))
        else:
            classifications.append(metadata_classify(row, patterns, usage_patterns))

    df = df.copy()
    df["verdict"] = [item.verdict for item in classifications]
    df["reason"] = [item.reason for item in classifications]
    df["matched_patterns"] = [
        " | ".join(item.matched_patterns) for item in classifications
    ]
    df["classifier_mode"] = mode
    return df


def output_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "display_name",
        "doi",
        "abstract",
        "verdict",
        "reason",
        "matched_patterns",
        "classifier_mode",
    ]
    return [column for column in preferred if column in df.columns]


def main() -> int:
    args = parse_args()
    try:
        metadata = load_metadata(args.metadata)
        target_title = normalize_space((metadata.get("target") or {}).get("display_name", ""))
        if not target_title:
            raise ValueError(f"No target.display_name found in {args.metadata}")

        if args.mode == "generated":
            patterns = compile_pattern_set(load_generated_patterns(args.regexp), "generated")
        else:
            patterns = compile_pattern_set(
                metadata_patterns(target_title, args.alias), "metadata"
            )

        df = pd.read_csv(args.input_csv, low_memory=False)
        classified = classify_dataframe(df, mode=args.mode, patterns=patterns)
        classified[output_columns(classified)].to_csv(args.output, index=False)
    except (OSError, ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Target: {target_title}")
    print(f"Mode: {args.mode}")
    if args.mode == "metadata":
        print(f"Aliases: {', '.join(args.alias) if args.alias else 'none'}")
    else:
        print(f"Regex file: {args.regexp}")
    print(f"Rows classified: {len(classified)}")
    print("\nClassification summary:")
    print(classified["verdict"].value_counts().to_string())
    print(f"\nSaved: {args.output}")

    dont_know = classified[classified["verdict"] == "dont_know"]
    if not dont_know.empty:
        print(f"\nSample dont_know titles ({len(dont_know)} total):")
        for title in dont_know["display_name"].head(10):
            print(f"  - {str(title)[:90]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
