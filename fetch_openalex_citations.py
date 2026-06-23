"""
Fetch papers that cite a target work from OpenAlex.

This is Step 1 from the project overview: a non-agent, static script that
downloads citing-paper metadata into a CSV that keywords.py can classify.

Examples:
    python fetch_openalex_citations.py 10.1038/sdata.2016.35
    python fetch_openalex_citations.py W2741809807 --output CitePaper.csv
    OPENALEX_API_KEY=... python fetch_openalex_citations.py doi:10.1038/sdata.2016.35
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


API_BASE_URL = "https://api.openalex.org"
DEFAULT_OUTPUT_CSV = "CitePaper.csv"
DEFAULT_PER_PAGE = 100
REQUEST_DELAY_SECONDS = 0.1

CSV_FIELDS = [
    "openalex_id",
    "display_name",
    "abstract",
    "doi",
    "type",
    "publication_year",
    "publication_date",
    "cited_by_count",
    "source_display_name",
    "openalex_url",
    "landing_page_url",
    "pdf_url",
    "best_oa_pdf_url",
    "oa_url",
    "is_oa",
    "content_pdf_url",
]


def normalize_work_identifier(value: str) -> str:
    """Return an OpenAlex /works/{id} identifier from DOI, DOI URL, or work ID."""
    value = value.strip()
    if not value:
        raise ValueError("Target work identifier cannot be empty.")

    lower_value = value.lower()
    if lower_value.startswith("https://openalex.org/"):
        return value.rstrip("/").split("/")[-1]
    if lower_value.startswith("openalex.org/"):
        return value.rstrip("/").split("/")[-1]
    if lower_value.startswith("https://doi.org/"):
        return "doi:" + value.split("doi.org/", 1)[1]
    if lower_value.startswith("doi.org/"):
        return "doi:" + value.split("doi.org/", 1)[1]
    if lower_value.startswith("doi:"):
        return value
    if lower_value.startswith("10."):
        return "doi:" + value
    return value


def openalex_short_id(openalex_id: str) -> str:
    """Convert https://openalex.org/W123 to W123; leave short IDs unchanged."""
    return openalex_id.rstrip("/").split("/")[-1]


def add_query_params(url: str, params: dict[str, str | int | None]) -> str:
    """Add or replace query parameters on a URL."""
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    for key, value in params.items():
        if value is None:
            query.pop(key, None)
        else:
            query[key] = str(value)
    return urlunparse(parsed._replace(query=urlencode(query)))


def request_json(url: str, *, timeout: int = 30) -> dict[str, Any]:
    """GET JSON from OpenAlex with a small user agent and useful errors."""
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "ReviewerProject/0.1 (OpenAlex citation fetcher)",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAlex HTTP {exc.code} for {url}\n{detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach OpenAlex for {url}: {exc.reason}") from exc


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """
    Convert OpenAlex's abstract_inverted_index into readable text.

    OpenAlex stores abstracts as {word: [positions]}; this reverses that mapping.
    """
    if not inverted_index:
        return ""

    positioned_words: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for position in positions:
            positioned_words[int(position)] = word

    if not positioned_words:
        return ""

    return " ".join(positioned_words[index] for index in sorted(positioned_words))


def source_name(work: dict[str, Any]) -> str:
    source = (work.get("primary_location") or {}).get("source") or {}
    return source.get("display_name") or ""


def landing_page_url(work: dict[str, Any]) -> str:
    return (work.get("primary_location") or {}).get("landing_page_url") or ""


def primary_pdf_url(work: dict[str, Any]) -> str:
    return (work.get("primary_location") or {}).get("pdf_url") or ""


def best_oa_pdf_url(work: dict[str, Any]) -> str:
    return (work.get("best_oa_location") or {}).get("pdf_url") or ""


def oa_url(work: dict[str, Any]) -> str:
    return (work.get("open_access") or {}).get("oa_url") or ""


def is_open_access(work: dict[str, Any]) -> bool:
    return bool((work.get("open_access") or {}).get("is_oa"))


def content_pdf_url(work: dict[str, Any]) -> str:
    return (work.get("content_urls") or {}).get("pdf") or ""


def row_from_work(work: dict[str, Any]) -> dict[str, Any]:
    ids = work.get("ids") or {}
    return {
        "openalex_id": work.get("id") or ids.get("openalex") or "",
        "display_name": work.get("display_name") or work.get("title") or "",
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "doi": work.get("doi") or ids.get("doi") or "",
        "type": work.get("type") or "",
        "publication_year": work.get("publication_year") or "",
        "publication_date": work.get("publication_date") or "",
        "cited_by_count": work.get("cited_by_count") or 0,
        "source_display_name": source_name(work),
        "openalex_url": work.get("id") or ids.get("openalex") or "",
        "landing_page_url": landing_page_url(work),
        "pdf_url": primary_pdf_url(work),
        "best_oa_pdf_url": best_oa_pdf_url(work),
        "oa_url": oa_url(work),
        "is_oa": is_open_access(work),
        "content_pdf_url": content_pdf_url(work),
    }


def resolve_target_work(target: str, api_key: str | None) -> dict[str, Any]:
    identifier = normalize_work_identifier(target)
    path_id = quote(identifier, safe=":")
    url = f"{API_BASE_URL}/works/{path_id}"
    url = add_query_params(url, {"api_key": api_key})
    return request_json(url)


def cited_by_url_for_work(work: dict[str, Any]) -> str:
    """
    Return the OpenAlex list URL for papers citing the target work.

    Older OpenAlex responses include cited_by_api_url. If it is absent, the
    equivalent list query is /works?filter=cites:{target_work_id}.
    """
    if work.get("cited_by_api_url"):
        return str(work["cited_by_api_url"])

    work_id = work.get("id") or (work.get("ids") or {}).get("openalex")
    if not work_id:
        raise ValueError("Target work response did not include an OpenAlex ID.")

    return f"{API_BASE_URL}/works?filter=cites:{openalex_short_id(work_id)}"


def iter_citing_works(
    first_url: str,
    *,
    api_key: str | None,
    per_page: int,
    max_results: int | None,
) -> tuple[list[dict[str, Any]], int | None]:
    rows: list[dict[str, Any]] = []
    cursor = "*"
    total_count: int | None = None

    select = ",".join(
        [
            "id",
            "ids",
            "doi",
            "display_name",
            "title",
            "abstract_inverted_index",
            "type",
            "publication_year",
            "publication_date",
            "cited_by_count",
            "primary_location",
            "best_oa_location",
            "open_access",
            "content_urls",
        ]
    )

    while True:
        url = add_query_params(
            first_url,
            {
                "api_key": api_key,
                "per_page": per_page,
                "cursor": cursor,
                "select": select,
            },
        )
        payload = request_json(url)
        meta = payload.get("meta") or {}
        total_count = meta.get("count", total_count)
        results = payload.get("results") or []

        for work in results:
            rows.append(row_from_work(work))
            if max_results is not None and len(rows) >= max_results:
                return rows, total_count

        next_cursor = meta.get("next_cursor")
        if not next_cursor or not results:
            return rows, total_count

        cursor = str(next_cursor)
        time.sleep(REQUEST_DELAY_SECONDS)


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def default_metadata_path(output_csv: str) -> str:
    output_path = Path(output_csv)
    return str(output_path.with_suffix(".meta.json"))


def metadata_from_fetch(
    *,
    requested_target: str,
    target_work: dict[str, Any],
    cited_by_url: str,
    output_csv: str,
    row_count: int,
    openalex_query_count: int | None,
    per_page: int,
    max_results: int | None,
) -> dict[str, Any]:
    ids = target_work.get("ids") or {}
    return {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_target": requested_target,
        "normalized_target": normalize_work_identifier(requested_target),
        "target": {
            "openalex_id": target_work.get("id") or ids.get("openalex") or "",
            "doi": target_work.get("doi") or ids.get("doi") or "",
            "display_name": target_work.get("display_name") or target_work.get("title") or "",
            "type": target_work.get("type") or "",
            "publication_year": target_work.get("publication_year") or "",
            "publication_date": target_work.get("publication_date") or "",
            "cited_by_count": target_work.get("cited_by_count"),
        },
        "query": {
            "api_base_url": API_BASE_URL,
            "cited_by_url": cited_by_url,
            "per_page": per_page,
            "max_results": max_results,
            "openalex_query_count": openalex_query_count,
        },
        "output": {
            "csv": output_csv,
            "rows": row_count,
            "columns": CSV_FIELDS,
        },
    }


def write_metadata(path: str, metadata: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch papers citing a target work from OpenAlex into a classifier-ready CSV.",
    )
    parser.add_argument(
        "target",
        help="Target OpenAlex work ID, DOI, DOI URL, or doi:... identifier.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENALEX_API_KEY"),
        help="OpenAlex API key. Defaults to OPENALEX_API_KEY.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=DEFAULT_PER_PAGE,
        help="Results per request. OpenAlex max is 100.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Stop after this many citing papers. Useful for smoke tests.",
    )
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Metadata JSON path. Default: output CSV path with .meta.json suffix.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.per_page < 1 or args.per_page > 100:
        print("--per-page must be between 1 and 100.", file=sys.stderr)
        return 2
    if args.max_results is not None and args.max_results < 1:
        print("--max-results must be at least 1.", file=sys.stderr)
        return 2

    try:
        target_work = resolve_target_work(args.target, args.api_key)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    target_id = target_work.get("id") or ""
    target_title = target_work.get("display_name") or target_work.get("title") or ""
    expected_count = target_work.get("cited_by_count")

    print(f"Target: {target_title}")
    print(f"OpenAlex ID: {target_id}")
    print(f"OpenAlex cited_by_count: {expected_count}")

    try:
        cited_by_url = cited_by_url_for_work(target_work)
        rows, total_count = iter_citing_works(
            cited_by_url,
            api_key=args.api_key,
            per_page=args.per_page,
            max_results=args.max_results,
        )
        write_csv(args.output, rows)
        metadata_path = args.metadata_output or default_metadata_path(args.output)
        write_metadata(
            metadata_path,
            metadata_from_fetch(
                requested_target=args.target,
                target_work=target_work,
                cited_by_url=cited_by_url,
                output_csv=args.output,
                row_count=len(rows),
                openalex_query_count=total_count,
                per_page=args.per_page,
                max_results=args.max_results,
            ),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Fetched citing works: {len(rows)}")
    if total_count is not None:
        print(f"OpenAlex query count: {total_count}")
    print(f"Saved: {args.output}")
    print(f"Saved metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
