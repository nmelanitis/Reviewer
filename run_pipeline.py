"""
Run the Reviewer pipeline steps 1-4, with optional cloud LLM add-ons.

The orchestrator intentionally calls the existing scripts as subprocesses. That
keeps each step independently usable while providing one command for common
workflows. Cloud LLM calls are opt-in via --llm-generate-regex,
--classifier llm, and --llm-write-review.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_REVIEW_OUTPUT = "review_prompt.md"
DEFAULT_FULLTEXT_REVIEW_OUTPUT = "review_prompt_fulltext.md"
DEFAULT_CITATION_CSV = "CitePaper.csv"
DEFAULT_PDF_DIR = "open-pdfs"
DEFAULT_FULLTEXT_DIR = "open-pdfs-text"
DEFAULT_LLM_REGEX_OUTPUT = "generated_keywords_regexp.py"
DEFAULT_LLM_REVIEW_OUTPUT = "review_draft.md"
DEFAULT_CLASSIFIED_CSV = "classified_papers.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Reviewer pipeline steps 1-4.")
    parser.add_argument("target", help="Target DOI, OpenAlex work ID, or DOI URL.")
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        help="Target-paper alias/keyword. Can be passed multiple times.",
    )
    parser.add_argument(
        "--mode",
        choices=["metadata", "generated"],
        default="metadata",
        help="Regex classification mode when --classifier regex. Default: metadata.",
    )
    parser.add_argument(
        "--classifier",
        choices=["regex", "llm"],
        default="regex",
        help="Classifier path. Default: regex.",
    )
    parser.add_argument(
        "--regexp",
        default=None,
        help="Trusted Python regex file for --mode generated.",
    )
    parser.add_argument(
        "--with-regex-context",
        action="store_true",
        help="Run Step 2 and create regex_context.md.",
    )
    parser.add_argument(
        "--with-pdfs",
        action="store_true",
        help="Download open PDFs after classification.",
    )
    parser.add_argument(
        "--citation-csv",
        default=DEFAULT_CITATION_CSV,
        help=f"Citation CSV path for Step 1 output and PDF download input. Default: {DEFAULT_CITATION_CSV}",
    )
    parser.add_argument(
        "--pdf-dir",
        default=DEFAULT_PDF_DIR,
        help=f"Directory for downloaded PDFs. Default: {DEFAULT_PDF_DIR}",
    )
    parser.add_argument(
        "--fulltext-dir",
        default=DEFAULT_FULLTEXT_DIR,
        help=f"Directory for extracted PDF text. Default: {DEFAULT_FULLTEXT_DIR}",
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=None,
        help="Maximum PDFs to download/extract.",
    )
    parser.add_argument(
        "--with-fulltext",
        action="store_true",
        help="Extract downloaded PDF text and build a full-text-aware prompt.",
    )
    parser.add_argument(
        "--review-output",
        default=DEFAULT_REVIEW_OUTPUT,
        help=f"Abstract-only review prompt output. Default: {DEFAULT_REVIEW_OUTPUT}",
    )
    parser.add_argument(
        "--classified-output",
        default=DEFAULT_CLASSIFIED_CSV,
        help=f"Classified CSV output used by review prompt steps. Default: {DEFAULT_CLASSIFIED_CSV}",
    )
    parser.add_argument(
        "--fulltext-review-output",
        default=DEFAULT_FULLTEXT_REVIEW_OUTPUT,
        help=(
            "Full-text-aware review prompt output. "
            f"Default: {DEFAULT_FULLTEXT_REVIEW_OUTPUT}"
        ),
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Pass through to fetch_openalex_citations.py for smoke tests.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for subprocesses. Default: current interpreter.",
    )
    parser.add_argument(
        "--extract-python",
        default=None,
        help="Python executable for PDF text extraction. Default: .venv/bin/python if present, else --python.",
    )
    parser.add_argument("--llm-provider", choices=["openai", "anthropic", "gemini"], default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--llm-generate-regex",
        action="store_true",
        help="Run Step 3C: generate regexes with a cloud LLM, then classify with them.",
    )
    parser.add_argument(
        "--llm-write-review",
        action="store_true",
        help="Run Step 5B: send the generated review prompt to a cloud LLM.",
    )
    parser.add_argument(
        "--llm-review-mode",
        choices=["single", "chunked", "selective"],
        default="selective",
        help="Review-writing LLM mode. Default: selective.",
    )
    parser.add_argument(
        "--llm-regex-output",
        default=DEFAULT_LLM_REGEX_OUTPUT,
        help=f"Cloud-generated regex file. Default: {DEFAULT_LLM_REGEX_OUTPUT}",
    )
    parser.add_argument(
        "--llm-review-output",
        default=DEFAULT_LLM_REVIEW_OUTPUT,
        help=f"Cloud-generated review draft. Default: {DEFAULT_LLM_REVIEW_OUTPUT}",
    )
    parser.add_argument(
        "--llm-max-input-chars",
        type=int,
        default=120000,
        help="Maximum prompt characters sent to cloud LLM. Default: 120000.",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=60000,
        help="Chunk size for --llm-review-mode chunked. Default: 60000.",
    )
    parser.add_argument(
        "--llm-classify-evidence",
        choices=["abstract", "fulltext"],
        default="abstract",
        help="Evidence for --classifier llm. Default: abstract.",
    )
    parser.add_argument(
        "--llm-classify-full-text-chars",
        type=int,
        default=30000,
        help="Maximum full-text characters per paper for --classifier llm. Default: 30000.",
    )
    parser.add_argument(
        "--llm-classify-output",
        default=None,
        help="Deprecated alias for --classified-output on the LLM classifier path.",
    )
    parser.add_argument(
        "--llm-classify-raw-dir",
        default=None,
        help="Optional directory for raw per-paper LLM classification responses.",
    )
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def add_alias_args(command: list[str], aliases: list[str]) -> list[str]:
    for alias in aliases:
        command.extend(["--alias", alias])
    return command


def extraction_python(args: argparse.Namespace) -> str:
    if args.extract_python:
        return args.extract_python
    venv_python = Path(".venv/bin/python")
    if venv_python.exists():
        return str(venv_python)
    return args.python


def metadata_path(args: argparse.Namespace) -> str:
    return str(Path(args.citation_csv).with_suffix(".meta.json"))


def download_pdfs(args: argparse.Namespace) -> None:
    pdf_command = [
        args.python,
        "download_open_pdfs.py",
        "--input",
        args.citation_csv,
        "--out-dir",
        args.pdf_dir,
    ]
    if args.max_pdfs is not None:
        pdf_command.extend(["--max-pdfs", str(args.max_pdfs)])
    run_step(pdf_command)


def extract_full_text(args: argparse.Namespace) -> None:
    extract_command = [
        extraction_python(args),
        "extract_pdf_text.py",
        "--manifest",
        str(Path(args.pdf_dir) / "manifest.csv"),
        "--out-dir",
        args.fulltext_dir,
    ]
    if args.max_pdfs is not None:
        extract_command.extend(["--max-pdfs", str(args.max_pdfs)])
    run_step(extract_command)


def build_review_prompt(args: argparse.Namespace, *, fulltext: bool = False) -> None:
    command = [
        args.python,
        "build_review_prompt.py",
        "--classified",
        args.classified_output,
    ]
    if fulltext:
        command.extend(
            [
                "--full-text-dir",
                args.fulltext_dir,
                "--output",
                args.fulltext_review_output,
            ]
        )
    else:
        command.extend(["--output", args.review_output])
    run_step(command)


def main() -> int:
    args = parse_args()
    if args.llm_classify_output:
        args.classified_output = args.llm_classify_output

    if args.classifier == "regex" and args.mode == "generated" and not args.regexp and not args.llm_generate_regex:
        print("Error: --regexp is required when --mode generated.", file=sys.stderr)
        return 2
    llm_fulltext_classifier = args.classifier == "llm" and args.llm_classify_evidence == "fulltext"
    if args.with_fulltext and not args.with_pdfs and not llm_fulltext_classifier:
        print("Error: --with-fulltext requires --with-pdfs.", file=sys.stderr)
        return 2
    if args.max_pdfs is not None and args.max_pdfs < 1:
        print("Error: --max-pdfs must be at least 1.", file=sys.stderr)
        return 2
    if args.max_results is not None and args.max_results < 1:
        print("Error: --max-results must be at least 1.", file=sys.stderr)
        return 2
    if (args.llm_generate_regex or args.llm_write_review) and (
        not args.llm_provider or not args.llm_model
    ):
        print("Error: --llm-provider and --llm-model are required for cloud LLM steps.", file=sys.stderr)
        return 2
    if args.classifier == "llm" and (not args.llm_provider or not args.llm_model):
        print("Error: --llm-provider and --llm-model are required for --classifier llm.", file=sys.stderr)
        return 2
    if args.classifier == "llm" and args.llm_generate_regex:
        print("Error: --llm-generate-regex is only valid with --classifier regex.", file=sys.stderr)
        return 2
    if args.llm_classify_full_text_chars < 200:
        print("Error: --llm-classify-full-text-chars must be at least 200.", file=sys.stderr)
        return 2

    try:
        fetch_command = [
            args.python,
            "fetch_openalex_citations.py",
            args.target,
            "--output",
            args.citation_csv,
        ]
        if args.max_results is not None:
            fetch_command.extend(["--max-results", str(args.max_results)])
        run_step(fetch_command)

        if args.with_regex_context or args.llm_generate_regex:
            context_command = [
                args.python,
                "prepare_regex_context.py",
                "--input-csv",
                args.citation_csv,
                "--metadata",
                metadata_path(args),
            ]
            run_step(add_alias_args(context_command, args.alias))

        fulltext_prepared_before_classification = False
        if args.classifier == "llm" and args.llm_classify_evidence == "fulltext":
            download_pdfs(args)
            extract_full_text(args)
            fulltext_prepared_before_classification = True

        if args.classifier == "llm":
            classify_command = [
                args.python,
                "llm_classify_papers.py",
                "--input-csv",
                args.citation_csv,
                "--metadata",
                metadata_path(args),
                "--output",
                args.classified_output,
                "--provider",
                args.llm_provider,
                "--model",
                args.llm_model,
                "--evidence-mode",
                args.llm_classify_evidence,
                "--full-text-dir",
                args.fulltext_dir,
                "--full-text-chars",
                str(args.llm_classify_full_text_chars),
            ]
            if args.max_results is not None:
                classify_command.extend(["--limit", str(args.max_results)])
            if args.llm_classify_raw_dir:
                classify_command.extend(["--raw-dir", args.llm_classify_raw_dir])
        elif args.llm_generate_regex:
            llm_regex_command = [
                args.python,
                "llm_generate_regex.py",
                "--provider",
                args.llm_provider,
                "--model",
                args.llm_model,
                "--output",
                args.llm_regex_output,
                "--max-input-chars",
                str(args.llm_max_input_chars),
            ]
            run_step(llm_regex_command)
            classify_command = [
                args.python,
                "keywords.py",
                "--input-csv",
                args.citation_csv,
                "--metadata",
                metadata_path(args),
                "--output",
                args.classified_output,
                "--mode",
                "generated",
                "--regexp",
                args.llm_regex_output,
            ]
        elif args.mode == "generated":
            classify_command = [
                args.python,
                "keywords.py",
                "--input-csv",
                args.citation_csv,
                "--metadata",
                metadata_path(args),
                "--output",
                args.classified_output,
                "--mode",
                "generated",
                "--regexp",
                args.regexp,
            ]
        else:
            classify_command = [
                args.python,
                "keywords.py",
                "--input-csv",
                args.citation_csv,
                "--metadata",
                metadata_path(args),
                "--output",
                args.classified_output,
                "--mode",
                args.mode,
            ]
            classify_command = add_alias_args(classify_command, args.alias)
        run_step(classify_command)

        build_review_prompt(args)

        if args.with_pdfs and not fulltext_prepared_before_classification:
            download_pdfs(args)

        if args.with_fulltext or fulltext_prepared_before_classification:
            if not fulltext_prepared_before_classification:
                extract_full_text(args)
            build_review_prompt(args, fulltext=True)

        if args.llm_write_review:
            prompt_path = (
                args.fulltext_review_output
                if args.with_fulltext or fulltext_prepared_before_classification
                else args.review_output
            )
            llm_review_command = [
                args.python,
                "llm_write_review.py",
                "--provider",
                args.llm_provider,
                "--model",
                args.llm_model,
                "--prompt",
                prompt_path,
                "--output",
                args.llm_review_output,
                "--mode",
                args.llm_review_mode,
                "--max-input-chars",
                str(args.llm_max_input_chars),
                "--chunk-chars",
                str(args.chunk_chars),
            ]
            run_step(llm_review_command)
    except subprocess.CalledProcessError as exc:
        print(f"\nPipeline failed at command: {' '.join(exc.cmd)}", file=sys.stderr)
        return exc.returncode or 1

    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
