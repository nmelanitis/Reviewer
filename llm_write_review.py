"""
Step 5B: ask a cloud LLM to write the final review draft.

The LLM sees Markdown prompt files produced by build_review_prompt.py. It does
not receive raw CSVs or PDFs. Whether it sees whole abstracts/full texts depends
on how build_review_prompt.py was configured with --abstract-chars and
--full-text-chars.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from llm_client import LLMConfig, generate_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a review draft with a cloud LLM.")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="review_prompt.md")
    parser.add_argument("--output", default="review_draft.md")
    parser.add_argument("--mode", choices=["single", "chunked", "selective"], default="selective")
    parser.add_argument("--max-input-chars", type=int, default=120000)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--chunk-chars", type=int, default=60000)
    parser.add_argument("--chunks-dir", default="review_chunks")
    return parser.parse_args()


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def call_llm(args: argparse.Namespace, prompt: str, *, max_output_tokens: int | None = None) -> str:
    return generate_text(
        LLMConfig(
            provider=args.provider,
            model=args.model,
            max_output_tokens=max_output_tokens or args.max_output_tokens,
        ),
        prompt,
    )


def ensure_size(text: str, max_chars: int, label: str) -> None:
    if len(text) > max_chars:
        raise ValueError(
            f"{label} is {len(text)} characters, above limit {max_chars}. "
            "Reduce build_review_prompt.py caps such as --max-yes, --abstract-chars, "
            "or --full-text-chars, or use --mode chunked."
        )


def review_instruction(prompt: str) -> str:
    return "\n\n".join(
        [
            "Write the final review section from the evidence below.",
            "Use cautious scholarly language. Do not invent details beyond the evidence.",
            "Focus on how the target paper/model was used, extended, adapted, or benchmarked.",
            prompt,
        ]
    )


def selective_prompt(prompt: str) -> str:
    """
    Compact prompt for cost/context control.

    build_review_prompt.py defaults to yes papers only. This function removes
    omitted-group bookkeeping and unavailable full-text lines while keeping the
    confirmed usage evidence.
    """
    lines = []
    skip_section = False
    for line in prompt.splitlines():
        if line.startswith("## Uncertain Usage Candidates") or line.startswith(
            "## Likely Background-Only / No Usage"
        ):
            skip_section = True
            continue
        if skip_section and line.startswith("## "):
            skip_section = False
        if skip_section:
            continue
        if "Full-text evidence: not available in extracted text set" in line:
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def split_chunks(text: str, chunk_chars: int) -> list[str]:
    if chunk_chars < 1000:
        raise ValueError("--chunk-chars must be at least 1000.")

    blocks = re.split(r"\n(?=\d+\. \*\*)", text)
    chunks: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n{block}".strip() if current else block.strip()
        if len(candidate) <= chunk_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(block) <= chunk_chars:
            current = block.strip()
        else:
            for index in range(0, len(block), chunk_chars):
                chunks.append(block[index : index + chunk_chars])
            current = ""
    if current:
        chunks.append(current)
    return chunks


def run_single(args: argparse.Namespace, prompt: str) -> str:
    final_prompt = review_instruction(prompt)
    ensure_size(final_prompt, args.max_input_chars, "Prompt")
    return call_llm(args, final_prompt)


def run_selective(args: argparse.Namespace, prompt: str) -> str:
    compact = selective_prompt(prompt)
    final_prompt = review_instruction(compact)
    ensure_size(final_prompt, args.max_input_chars, "Selective prompt")
    return call_llm(args, final_prompt)


def run_chunked(args: argparse.Namespace, prompt: str) -> str:
    chunks_dir = Path(args.chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks = split_chunks(prompt, args.chunk_chars)
    summaries = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_prompt = "\n\n".join(
            [
                "Summarize this evidence chunk for a later literature-review synthesis.",
                "Focus on how papers use, extend, adapt, or benchmark the target paper/model.",
                "Preserve paper titles and cautious uncertainty notes.",
                chunk,
            ]
        )
        ensure_size(chunk_prompt, args.max_input_chars, f"Chunk {index}")
        summary = call_llm(args, chunk_prompt)
        summary_path = chunks_dir / f"chunk_{index:03d}_summary.md"
        summary_path.write_text(summary, encoding="utf-8")
        summaries.append(f"## Chunk {index}\n\n{summary}")

    final_prompt = "\n\n".join(
        [
            "Write the final review section from these chunk summaries.",
            "Focus on confirmed target-paper/model usage in the literature.",
            "Use cautious scholarly language and do not invent details.",
            *summaries,
        ]
    )
    ensure_size(final_prompt, args.max_input_chars, "Final synthesis prompt")
    (chunks_dir / "final_review_prompt.md").write_text(final_prompt, encoding="utf-8")
    return call_llm(args, final_prompt)


def main() -> int:
    args = parse_args()
    try:
        prompt = read_text(args.prompt)
        if args.mode == "single":
            review = run_single(args, prompt)
        elif args.mode == "chunked":
            review = run_chunked(args, prompt)
        else:
            review = run_selective(args, prompt)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    Path(args.output).write_text(review.strip() + "\n", encoding="utf-8")
    print(f"Saved review draft: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
