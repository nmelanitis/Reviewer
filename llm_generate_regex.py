"""
Step 3C: ask a cloud LLM to generate regex lists for keywords.py.

The script sends paper_usage_regex_prompt.md plus regex_context.md to the
selected provider, extracts a Python code block from the response, validates
the expected regex-list variables, and writes a trusted local regexp file.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any

from llm_client import LLMConfig, generate_text


EXPECTED_LISTS = ["STRONG_YES", "WEAK_YES", "BACKGROUND_ONLY", "AMBIGUOUS"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate regex lists with a cloud LLM.")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="paper_usage_regex_prompt.md")
    parser.add_argument("--context", default="regex_context.md")
    parser.add_argument("--output", default="generated_keywords_regexp.py")
    parser.add_argument("--raw-output", default=None, help="Optional path for the raw LLM response.")
    parser.add_argument("--max-input-chars", type=int, default=120000)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    return parser.parse_args()


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_prompt(instructions: str, context: str) -> str:
    return "\n\n".join(
        [
            instructions,
            "Return only one fenced Python code block containing these variables:",
            ", ".join(EXPECTED_LISTS),
            "Here is the regex context:",
            context,
        ]
    )


def extract_python_code(response_text: str) -> str:
    python_blocks = re.findall(r"```(?:python|py)\s*(.*?)```", response_text, re.S | re.I)
    if python_blocks:
        return python_blocks[0].strip()

    generic_blocks = re.findall(r"```\s*(.*?)```", response_text, re.S)
    if generic_blocks:
        return generic_blocks[0].strip()

    return response_text.strip()


def validate_regex_file(code: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"LLM output is not valid Python: {exc}") from exc

    namespace: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in EXPECTED_LISTS:
                namespace[target.id] = ast.literal_eval(node.value)

    for name in EXPECTED_LISTS:
        value = namespace.get(name, [])
        if not isinstance(value, list):
            raise ValueError(f"{name} must be a list of regex strings.")
        for pattern in value:
            if not isinstance(pattern, str):
                raise ValueError(f"{name} contains a non-string item: {pattern!r}")
            re.compile(pattern)

    if not namespace.get("STRONG_YES") and not namespace.get("WEAK_YES"):
        raise ValueError("LLM output must define STRONG_YES and/or WEAK_YES patterns.")

    return namespace


def normalized_regex_file(code: str) -> str:
    validate_regex_file(code)
    header = [
        '"""Generated regex lists for keywords.py.',
        "",
        "Created by llm_generate_regex.py. Review before relying on results.",
        '"""',
        "",
    ]
    return "\n".join(header) + code.strip() + "\n"


def main() -> int:
    args = parse_args()
    instructions = read_text(args.prompt)
    context = read_text(args.context)
    prompt = build_prompt(instructions, context)
    if len(prompt) > args.max_input_chars:
        print(
            f"Prompt is {len(prompt)} characters, above --max-input-chars={args.max_input_chars}. "
            "Regenerate regex_context.md with fewer examples/snippets."
        )
        return 2

    try:
        response = generate_text(
            LLMConfig(
                provider=args.provider,
                model=args.model,
                max_output_tokens=args.max_output_tokens,
            ),
            prompt,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    if args.raw_output:
        Path(args.raw_output).write_text(response, encoding="utf-8")

    code = extract_python_code(response)
    try:
        output = normalized_regex_file(code)
    except (ValueError, re.error) as exc:
        print(f"Could not validate LLM regex output: {exc}")
        if not args.raw_output:
            print("Tip: rerun with --raw-output llm_regex_raw.md to inspect the response.")
        return 1

    Path(args.output).write_text(output, encoding="utf-8")
    print(f"Saved regex file: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
