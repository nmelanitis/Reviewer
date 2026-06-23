"""
Extract text from downloaded PDFs into open-pdfs-text/.

Step 4b turns PDFs from download_open_pdfs.py into plain text files that can
feed build_review_prompt.py --full-text-dir. It tries available extractors in
this order:
    1. PyMuPDF (import fitz)
    2. pypdf
    3. PyPDF2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable


DEFAULT_MANIFEST = "open-pdfs/manifest.csv"
DEFAULT_OUT_DIR = "open-pdfs-text"

MANIFEST_FIELDS = [
    "openalex_id",
    "display_name",
    "doi",
    "pdf_path",
    "text_path",
    "status",
    "extractor",
    "characters",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract text from downloaded PDFs.")
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"PDF download manifest. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for .txt files. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=None,
        help="Stop after this many PDFs are extracted or skipped.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even when text file already exists.",
    )
    return parser.parse_args()


def load_manifest(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_text_filename(row: dict[str, str]) -> str:
    pdf_path = Path(row.get("local_path") or row.get("pdf_path") or "")
    if pdf_path.name:
        return pdf_path.with_suffix(".txt").name
    openalex_id = (row.get("openalex_id") or "paper").rstrip("/").split("/")[-1]
    return f"{openalex_id}.txt"


def extract_with_pymupdf(path: Path) -> str:
    import fitz

    parts: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    return "\n".join(parts)


def extract_with_pypdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_with_pypdf2(path: Path) -> str:
    from PyPDF2 import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def available_extractors() -> list[tuple[str, Callable[[Path], str]]]:
    extractors: list[tuple[str, Callable[[Path], str]]] = []
    try:
        import fitz  # noqa: F401

        extractors.append(("PyMuPDF", extract_with_pymupdf))
    except ImportError:
        pass
    try:
        import pypdf  # noqa: F401

        extractors.append(("pypdf", extract_with_pypdf))
    except ImportError:
        pass
    try:
        import PyPDF2  # noqa: F401

        extractors.append(("PyPDF2", extract_with_pypdf2))
    except ImportError:
        pass
    return extractors


def extract_text(path: Path, extractors: list[tuple[str, Callable[[Path], str]]]) -> tuple[str, str]:
    errors = []
    for name, extractor in extractors:
        try:
            text = extractor(path).strip()
            if text:
                return text, name
            errors.append(f"{name}: no text extracted")
        except Exception as exc:  # PDF parsers raise varied exceptions.
            errors.append(f"{name}: {exc}")
    raise RuntimeError("; ".join(errors) or "No extractors available")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def result_row(
    row: dict[str, str],
    *,
    pdf_path: str,
    text_path: str,
    status: str,
    extractor: str = "",
    characters: int = 0,
    error: str = "",
) -> dict[str, str]:
    return {
        "openalex_id": row.get("openalex_id", ""),
        "display_name": row.get("display_name", ""),
        "doi": row.get("doi", ""),
        "pdf_path": pdf_path,
        "text_path": text_path,
        "status": status,
        "extractor": extractor,
        "characters": str(characters),
        "error": error,
    }


def main() -> int:
    args = parse_args()
    if args.max_pdfs is not None and args.max_pdfs < 1:
        print("--max-pdfs must be at least 1.")
        return 2

    extractors = available_extractors()
    if not extractors:
        print("No PDF extraction libraries available. Install PyMuPDF, pypdf, or PyPDF2.")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest)
    results: list[dict[str, str]] = []
    completed = 0

    for row in rows:
        if row.get("status") not in {"downloaded", "skipped_exists"}:
            continue

        pdf_path = Path(row.get("local_path", ""))
        text_path = out_dir / safe_text_filename(row)
        if not pdf_path.exists():
            results.append(
                result_row(
                    row,
                    pdf_path=str(pdf_path),
                    text_path=str(text_path),
                    status="missing_pdf",
                    error="PDF file not found",
                )
            )
            continue

        if text_path.exists() and not args.force:
            characters = len(text_path.read_text(encoding="utf-8", errors="replace"))
            results.append(
                result_row(
                    row,
                    pdf_path=str(pdf_path),
                    text_path=str(text_path),
                    status="skipped_exists",
                    characters=characters,
                )
            )
            completed += 1
        else:
            try:
                text, extractor = extract_text(pdf_path, extractors)
                text_path.write_text(text + "\n", encoding="utf-8")
                results.append(
                    result_row(
                        row,
                        pdf_path=str(pdf_path),
                        text_path=str(text_path),
                        status="extracted",
                        extractor=extractor,
                        characters=len(text),
                    )
                )
                completed += 1
            except Exception as exc:
                results.append(
                    result_row(
                        row,
                        pdf_path=str(pdf_path),
                        text_path=str(text_path),
                        status="failed",
                        error=str(exc),
                    )
                )

        if args.max_pdfs is not None and completed >= args.max_pdfs:
            break

    manifest_path = out_dir / "manifest.csv"
    write_manifest(manifest_path, results)
    status_counts: dict[str, int] = {}
    for item in results:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

    print("Extractors: " + ", ".join(name for name, _ in extractors))
    print(f"PDFs processed: {len(results)}")
    print("Statuses: " + ", ".join(f"{key}={value}" for key, value in status_counts.items()))
    print(f"Saved manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
