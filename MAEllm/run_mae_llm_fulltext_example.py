"""
Run the MAE direct-LLM full-text classification example from MAEllm/config.json.

This runner mirrors the MAEex reproducibility style, but skips regex generation
and uses llm_classify_papers.py --evidence-mode fulltext. It stops after
review_prompt_fulltext.md and does not call llm_write_review.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MAE direct LLM full-text example.")
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
        "classified_csv": out_dir / "classified_papers.csv",
        "pdf_dir": out_dir / "open-pdfs",
        "fulltext_dir": out_dir / "open-pdfs-text",
        "review_prompt_fulltext": out_dir / "review_prompt_fulltext.md",
    }


def venv_python() -> str:
    python_path = REPO_ROOT / ".venv" / "bin" / "python"
    if not python_path.exists():
        raise FileNotFoundError(
            f"Expected virtualenv Python not found: {python_path}. "
            "Create it with `python3 -m venv .venv` and install project dependencies."
        )
    return str(python_path)


def add_optional_int(command: list[str], flag: str, item: Any) -> None:
    if item is not None:
        command.extend([flag, str(item)])


def write_run_header(log_path: Path, config_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Started: {datetime.now().isoformat()}\n")
        handle.write(f"Config: {config_path}\n")
        handle.write(f"Dry run: {dry_run}\n")
        handle.write("Secrets are not printed; API keys are read from .env by llm_client.py.\n")
        handle.write("This runner stops after review_prompt_fulltext.md.\n")


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


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    out_dir = (REPO_ROOT / value(config, "output_dir", "MAEllm")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(out_dir)
    paths["llm_raw_dir"] = out_dir / str(value(config, "llm_classify.raw_dir", "llm-classification-raw"))
    log_path = out_dir / "run.log"
    write_run_header(log_path, config_path, args.dry_run)

    provider = value(config, "provider", "openai")
    model = value(config, "model", "gpt-5.4-mini")
    target = value(config, "target", "10.48550/arxiv.2111.06377")
    python_exe = venv_python()

    print("Effective core config:")
    print(f"  target: {target}")
    print(f"  output_dir: {out_dir}")
    print(f"  provider/model: {provider}/{model}")
    print(f"  python: {python_exe}")
    print("  stops after: review_prompt_fulltext.md")
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

        classify_command = [
            python_exe,
            "llm_classify_papers.py",
            "--input-csv",
            str(paths["citation_csv"]),
            "--metadata",
            str(paths["metadata_json"]),
            "--output",
            str(paths["classified_csv"]),
            "--provider",
            provider,
            "--model",
            model,
            "--evidence-mode",
            str(value(config, "llm_classify.evidence_mode", "fulltext")),
            "--full-text-dir",
            str(paths["fulltext_dir"]),
            "--abstract-chars",
            str(value(config, "llm_classify.abstract_chars", "all")),
            "--full-text-chars",
            str(value(config, "llm_classify.full_text_chars", 30000)),
            "--max-output-tokens",
            str(value(config, "llm_classify.max_output_tokens", 800)),
            "--retries",
            str(value(config, "llm_classify.retries", 3)),
            "--retry-sleep-seconds",
            str(value(config, "llm_classify.retry_sleep_seconds", 30)),
            "--raw-dir",
            str(paths["llm_raw_dir"]),
        ]
        if bool_value(config, "llm_classify.resume", True):
            classify_command.append("--resume")
        run_command(classify_command, log_path=log_path, dry_run=args.dry_run)

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
    except subprocess.CalledProcessError as exc:
        print(f"\nRun failed at command: {' '.join(exc.cmd)}")
        print(f"See log: {log_path}")
        return exc.returncode or 1

    print("\nMAEllm example complete.")
    print(f"Review prompt: {paths['review_prompt_fulltext']}")
    print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
