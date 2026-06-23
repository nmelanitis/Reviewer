"""
Run the full MAE citation-review example from MAEex/config.json.

This runner is intentionally explicit rather than clever. It keeps generated
files under MAEex/, calls the existing project scripts with concrete paths, and
logs each command for reproducibility. Cloud calls use the provider/model in the
config and read API keys through llm_client.py from the repo-level .env file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"
DEFAULT_LOG = SCRIPT_DIR / "run.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MAE full-text Gemini example.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Config JSON path. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing network/API/PDF steps.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def value(config: dict[str, Any], key: str, fallback: Any) -> Any:
    current: Any = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return fallback
        current = current[part]
    return current


def bool_value(config: dict[str, Any], key: str, fallback: bool) -> bool:
    return bool(value(config, key, fallback))


def output_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "citation_csv": out_dir / "CitePaper.csv",
        "metadata_json": out_dir / "CitePaper.meta.json",
        "regex_context": out_dir / "regex_context.md",
        "generated_regex": out_dir / "generated_keywords_regexp.py",
        "llm_regex_raw": out_dir / "llm_regex_raw.md",
        "classified_csv": out_dir / "classified_papers.csv",
        "pdf_dir": out_dir / "open-pdfs",
        "fulltext_dir": out_dir / "open-pdfs-text",
        "review_prompt_fulltext": out_dir / "review_prompt_fulltext.md",
        "review_chunks": out_dir / "review_chunks",
        "review_draft": out_dir / "review_draft.md",
    }


def venv_python() -> str:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise FileNotFoundError(
            f"Expected virtualenv Python not found: {venv_python}. "
            "Create it with `python3 -m venv .venv` and install project dependencies."
        )
    return str(venv_python)


def add_aliases(command: list[str], aliases: list[str]) -> list[str]:
    for alias in aliases:
        command.extend(["--alias", alias])
    return command


def add_optional_int(command: list[str], flag: str, item: Any) -> None:
    if item is not None:
        command.extend([flag, str(item)])


def log_line(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def run_command(command: list[str], *, log_path: Path, dry_run: bool) -> None:
    printable = " ".join(command)
    print("\n$ " + printable, flush=True)
    log_line(log_path, "\n$ " + printable)
    if dry_run:
        return

    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )


def run_command_with_retries(
    command: list[str],
    *,
    log_path: Path,
    dry_run: bool,
    retries: int,
    sleep_seconds: int,
) -> None:
    attempts = max(retries, 0) + 1
    for attempt in range(1, attempts + 1):
        try:
            run_command(command, log_path=log_path, dry_run=dry_run)
            return
        except subprocess.CalledProcessError:
            if attempt >= attempts:
                raise
            message = (
                f"Command failed on attempt {attempt}/{attempts}; "
                f"retrying in {sleep_seconds} seconds."
            )
            print(message, flush=True)
            log_line(log_path, message)
            time.sleep(sleep_seconds)


def write_run_header(log_path: Path, config_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Started: {datetime.now().isoformat()}\n")
        handle.write(f"Config: {config_path}\n")
        handle.write(f"Dry run: {dry_run}\n")
        handle.write("Secrets are not printed; API keys are read from .env by llm_client.py.\n")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    out_dir = (REPO_ROOT / value(config, "output_dir", "MAEex")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(out_dir)
    log_path = out_dir / "run.log"
    write_run_header(log_path, config_path, args.dry_run)

    provider = value(config, "provider", "gemini")
    model = value(config, "model", "gemini-2.5-flash-lite")
    target = value(config, "target", "10.48550/arxiv.2111.06377")
    aliases = list(value(config, "aliases", []))
    python_exe = venv_python()

    print("Effective core config:")
    print(f"  target: {target}")
    print(f"  output_dir: {out_dir}")
    print(f"  provider/model: {provider}/{model}")
    print(f"  python: {python_exe}")
    print(f"  aliases: {', '.join(aliases) or '(none)'}")
    print("Fallback behavior is documented in config.json under fallbacks_when_unspecified.")

    try:
        fetch_command = [
            python_exe,
            "fetch_openalex_citations.py",
            target,
            "--output",
            str(paths["citation_csv"]),
            "--metadata-output",
            str(paths["metadata_json"]),
            "--per-page",
            str(value(config, "fetch.per_page", 100)),
        ]
        add_optional_int(fetch_command, "--max-results", value(config, "fetch.max_results", None))
        run_command(fetch_command, log_path=log_path, dry_run=args.dry_run)

        regex_context_command = [
            python_exe,
            "prepare_regex_context.py",
            "--input-csv",
            str(paths["citation_csv"]),
            "--metadata",
            str(paths["metadata_json"]),
            "--output",
            str(paths["regex_context"]),
            "--max-examples",
            str(value(config, "regex_context.max_examples", 8)),
            "--snippet-chars",
            str(value(config, "regex_context.snippet_chars", 700)),
        ]
        run_command(add_aliases(regex_context_command, aliases), log_path=log_path, dry_run=args.dry_run)

        llm_regex_command = [
            python_exe,
            "llm_generate_regex.py",
            "--provider",
            provider,
            "--model",
            model,
            "--prompt",
            "paper_usage_regex_prompt.md",
            "--context",
            str(paths["regex_context"]),
            "--output",
            str(paths["generated_regex"]),
            "--raw-output",
            str(paths["llm_regex_raw"]),
            "--max-input-chars",
            str(value(config, "llm_regex.max_input_chars", 250000)),
            "--max-output-tokens",
            str(value(config, "llm_regex.max_output_tokens", 8192)),
        ]
        run_command_with_retries(
            llm_regex_command,
            log_path=log_path,
            dry_run=args.dry_run,
            retries=int(value(config, "llm_regex.retries", 3)),
            sleep_seconds=int(value(config, "llm_regex.retry_sleep_seconds", 30)),
        )

        classify_command = [
            python_exe,
            "keywords.py",
            "--input-csv",
            str(paths["citation_csv"]),
            "--metadata",
            str(paths["metadata_json"]),
            "--output",
            str(paths["classified_csv"]),
            "--mode",
            "generated",
            "--regexp",
            str(paths["generated_regex"]),
        ]
        run_command(classify_command, log_path=log_path, dry_run=args.dry_run)

        pdf_command = [
            python_exe,
            "download_open_pdfs.py",
            "--input",
            str(paths["citation_csv"]),
            "--out-dir",
            str(paths["pdf_dir"]),
            "--timeout",
            str(value(config, "pdfs.timeout_seconds", 45)),
        ]
        add_optional_int(pdf_command, "--max-pdfs", value(config, "pdfs.max_pdfs", None))
        if bool_value(config, "pdfs.force_download", False):
            pdf_command.append("--force")
        run_command(pdf_command, log_path=log_path, dry_run=args.dry_run)

        extract_command = [
            python_exe,
            "extract_pdf_text.py",
            "--manifest",
            str(paths["pdf_dir"] / "manifest.csv"),
            "--out-dir",
            str(paths["fulltext_dir"]),
        ]
        add_optional_int(extract_command, "--max-pdfs", value(config, "extract_text.max_pdfs", None))
        if bool_value(config, "extract_text.force_extract", False):
            extract_command.append("--force")
        run_command(extract_command, log_path=log_path, dry_run=args.dry_run)

        review_prompt_command = [
            python_exe,
            "build_review_prompt.py",
            "--classified",
            str(paths["classified_csv"]),
            "--metadata",
            str(paths["metadata_json"]),
            "--output",
            str(paths["review_prompt_fulltext"]),
            "--max-yes",
            str(value(config, "review_prompt.max_yes", 999999)),
            "--max-dont-know",
            str(value(config, "review_prompt.max_dont_know", 0)),
            "--max-no",
            str(value(config, "review_prompt.max_no", 0)),
            "--abstract-chars",
            str(value(config, "review_prompt.abstract_chars", "all")),
            "--full-text-dir",
            str(paths["fulltext_dir"]),
            "--full-text-chars",
            str(value(config, "review_prompt.full_text_chars", "all")),
        ]
        if bool_value(config, "review_prompt.include_dont_know", False):
            review_prompt_command.append("--include-dont-know")
        if bool_value(config, "review_prompt.include_no", False):
            review_prompt_command.append("--include-no")
        run_command(review_prompt_command, log_path=log_path, dry_run=args.dry_run)

        llm_review_command = [
            python_exe,
            "llm_write_review.py",
            "--provider",
            provider,
            "--model",
            model,
            "--mode",
            str(value(config, "llm_review.mode", "chunked")),
            "--prompt",
            str(paths["review_prompt_fulltext"]),
            "--output",
            str(paths["review_draft"]),
            "--max-input-chars",
            str(value(config, "llm_review.max_input_chars", 3000000)),
            "--max-output-tokens",
            str(value(config, "llm_review.max_output_tokens", 65536)),
            "--chunk-chars",
            str(value(config, "llm_review.chunk_chars", 60000)),
            "--chunks-dir",
            str(paths["review_chunks"]),
        ]
        run_command_with_retries(
            llm_review_command,
            log_path=log_path,
            dry_run=args.dry_run,
            retries=int(value(config, "llm_review.retries", 3)),
            sleep_seconds=int(value(config, "llm_review.retry_sleep_seconds", 30)),
        )
    except subprocess.CalledProcessError as exc:
        print(f"\nCommand failed: {' '.join(exc.cmd)}")
        print(f"See log: {log_path}")
        return exc.returncode or 1

    print(f"\nDone. Log: {log_path}")
    if args.dry_run:
        print("Dry run only; no network/API/PDF work was executed.")
    else:
        print(f"Final review draft: {paths['review_draft']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
