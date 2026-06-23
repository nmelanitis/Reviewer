# Paper-Usage Regex Generation Prompt

You are helping build a Python `re`-based classifier for a citation review workflow.

The user has fetched papers that cite one target paper. Your task is to propose regex
rules and classification logic that detect whether each citing paper actually uses,
extends, applies, benchmarks, reproduces, adapts, or centrally discusses the target
paper/method.

Use the accompanying `regex_context.md` content as your evidence. Do not assume the
target paper is always the same paper; use the target metadata and vocabulary seeds
provided in that context.

## Labels

- `yes`: the citing paper clearly uses, applies, extends, benchmarks, reproduces,
  adapts, or centrally discusses the target paper/method.
- `no`: the citing paper only cites the target in background/related work, or has no
  target-specific evidence in title/abstract.
- `dont_know`: the title/abstract is missing, ambiguous, or does not provide enough
  evidence about the citation role.

## Output Requirements

Return:

1. Python-compatible regex lists grouped by purpose:
   - strong `yes` evidence
   - weak `yes` evidence
   - `no` / background-only evidence if useful
   - ambiguity markers if useful
2. A decision priority for combining those regexes.
3. A short explanation of the precision/recall tradeoff.
4. At least five examples from the context and how your rules classify them.
5. Notes about aliases or target-specific terms the human should add manually.

Prefer high precision over high recall for `yes`. A false positive is worse than a
`dont_know`, because `dont_know` can be reviewed manually.

Keep the regexes suitable for Python's built-in `re` module. Avoid lookbehinds or
engine-specific features.
