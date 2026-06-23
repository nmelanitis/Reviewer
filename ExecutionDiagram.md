# Execution Diagrams

This file shows the main Reviewer execution paths. The paths are intentionally
different because regex classification uses abstracts/titles/metadata, while
direct LLM full-text classification needs extracted full text before it can
classify papers.

## 1. Regex Classification, Abstract-Only Review

```text
Target DOI
   |
   v
fetch_openalex_citations.py
   |
   v
CitePaper.csv + CitePaper.meta.json
   |
   v
prepare_regex_context.py
   |
   v
regex_context.md
   |
   v
llm_generate_regex.py
   |
   v
generated_keywords_regexp.py
   |
   v
keywords.py --mode generated
   |
   v
classified_papers.csv
   |
   v
build_review_prompt.py
   |
   v
review_prompt.md
   |
   v
llm_write_review.py
   |
   v
review_draft.md
```

In this path, regexes are generated from `regex_context.md`, which is built
from citation metadata, titles, abstracts, and user aliases. Full texts are not
used.

## 2. Regex Classification, Full-Text Review

```text
Target DOI
   |
   v
fetch_openalex_citations.py
   |
   v
CitePaper.csv + CitePaper.meta.json
   |
   v
prepare_regex_context.py
   |
   v
regex_context.md
   |
   v
llm_generate_regex.py
   |
   v
generated_keywords_regexp.py
   |
   v
keywords.py --mode generated
   |
   v
classified_papers.csv
   |
   v
build_review_prompt.py
   |
   v
review_prompt.md
   |
   v
download_open_pdfs.py
   |
   v
open-pdfs/
   |
   v
extract_pdf_text.py
   |
   v
open-pdfs-text/
   |
   v
build_review_prompt.py --full-text-dir open-pdfs-text
   |
   v
review_prompt_fulltext.md
   |
   v
llm_write_review.py
   |
   v
review_draft_fulltext.md
```

In the orchestrator, regex classification happens before PDF download/extraction.
That is correct because the regex classifier does not read full texts. The
orchestrator first builds an abstract-only prompt, then downloads/extracts PDFs
and builds the full-text prompt when `--with-fulltext` is enabled.

## 3. Direct LLM Classification, Abstract-Only Review

```text
Target DOI
   |
   v
fetch_openalex_citations.py
   |
   v
CitePaper.csv + CitePaper.meta.json
   |
   v
llm_classify_papers.py --evidence-mode abstract
   |
   v
classified_papers.csv
   |
   v
build_review_prompt.py
   |
   v
review_prompt.md
   |
   v
llm_write_review.py
   |
   v
review_draft.md
```

This path skips regex generation and regex matching. The LLM classifies each
citing paper directly from its title, metadata, and abstract.

## 4. Direct LLM Classification, Full-Text Review

```text
Target DOI
   |
   v
fetch_openalex_citations.py
   |
   v
CitePaper.csv + CitePaper.meta.json
   |
   v
download_open_pdfs.py
   |
   v
open-pdfs/
   |
   v
extract_pdf_text.py
   |
   v
open-pdfs-text/
   |
   v
llm_classify_papers.py --evidence-mode fulltext
   |
   v
classified_papers.csv
   |
   v
build_review_prompt.py --full-text-dir open-pdfs-text
   |
   v
review_prompt_fulltext.md
   |
   v
llm_write_review.py
   |
   v
review_draft_fulltext.md
```

In this path, PDF download and text extraction must happen before
classification because `llm_classify_papers.py --evidence-mode fulltext` reads
the extracted text. If extracted text is unavailable for a paper, the classifier
falls back to the abstract and records `evidence_source=abstract_fallback`.

## Key Difference

```text
Regex path:
classification evidence = title + abstract + metadata + aliases
full text evidence = optional, review-writing only

Direct LLM full-text path:
classification evidence = title + abstract + extracted full text when available
full text evidence = required before classification
```
