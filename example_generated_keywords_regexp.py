"""
Example generated-regex file for keywords.py.

Use with:
    python3 keywords.py --mode generated --regexp example_generated_keywords_regexp.py

In real use, replace these lists with regexes generated from
paper_usage_regex_prompt.md + regex_context.md.
"""

STRONG_YES = [
    r"\bmasked autoencoders?\b",
    r"\bMAE\b",
]

WEAK_YES = [
    r"\bmasked image modeling\b",
]

BACKGROUND_ONLY = [
    r"\bsurvey\b",
]

AMBIGUOUS = [
    r"\bself-supervised\b",
]
