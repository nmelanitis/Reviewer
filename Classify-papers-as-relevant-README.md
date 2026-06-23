# How Papers Are Classified As Relevant: Prompt Creation And LLM Regex Generation

This note explains how the project classifies citing papers into:

- `yes`: the citing paper appears to use, apply, extend, benchmark, reproduce,
  adapt, or centrally discuss the target paper.
- `no`: the citing paper appears to mention the target only as background, or
  has no target-specific evidence in the title/abstract.
- `dont_know`: the evidence is missing, ambiguous, or not enough to decide.

The generated-regex workflow has three main steps:

```text
CitePaper.csv + CitePaper.meta.json
        |
        v
prepare_regex_context.py -> regex_context.md
        |
        v
paper_usage_regex_prompt.md + regex_context.md
        |
        v
llm_generate_regex.py -> generated_keywords_regexp.py
        |
        v
keywords.py --mode generated -> classified_papers.csv
```

There is also a direct LLM classifier path:

```text
CitePaper.csv + CitePaper.meta.json
        |
        v
llm_classify_papers.py -> classified_papers.csv
```

The direct LLM path skips `prepare_regex_context.py`,
`paper_usage_regex_prompt.md`, `llm_generate_regex.py`, and regex matching. It
asks a cloud LLM to classify each paper directly from its abstract or extracted
full text.

## 1. What `prepare_regex_context.py` Reads

`prepare_regex_context.py` creates the evidence context used to ask an LLM for
regular expressions. It does not make the final classification itself.

> Important: the regular expression is built using information from the
> abstracts of the citing papers, together with titles, target metadata, and
> user-provided aliases.

By default it reads:

```text
CitePaper.csv
CitePaper.meta.json
```

You can also pass explicit paths:

```bash
python3 prepare_regex_context.py \
  --input-csv MAEex/CitePaper.csv \
  --metadata MAEex/CitePaper.meta.json \
  --output MAEex/regex_context.md \
  --alias MAE \
  --alias "masked autoencoder"
```

It parses three kinds of resources.

Target metadata from `CitePaper.meta.json`:

- target title
- DOI
- OpenAlex ID

Citing-paper rows from `CitePaper.csv`:

- title / `display_name`
- abstract
- DOI
- paper type
- publication year
- cited-by count

Optional aliases passed by the user:

- `MAE`
- `masked autoencoder`
- `masked autoencoders`
- any other target-specific terms useful for the paper under review

Important detail: the script reads all rows in the citation CSV, but
`regex_context.md` does not necessarily include every abstract in full. It
includes capped examples from heuristic buckets. The cap is controlled by:

```bash
--max-examples
--snippet-chars
```

For the MAE example we used:

```bash
python3 prepare_regex_context.py \
  --max-examples 12 \
  --snippet-chars 2500 \
  --alias MAE \
  --alias "masked autoencoder" \
  --alias "masked autoencoders"
```

## 2. What `prepare_regex_context.py` Does

First, it extracts the target title from metadata, unless `--target-title` is
provided.

For the MAE example, the target title is:

```text
Masked Autoencoders Are Scalable Vision Learners
```

Then it tokenizes the title, removes stopwords, and creates:

- title terms
- title phrases

Relevant code:

- [prepare_regex_context.py](/Users/nme/Documents/pythonProjects/Reviewer/prepare_regex_context.py:160)
- [prepare_regex_context.py](/Users/nme/Documents/pythonProjects/Reviewer/prepare_regex_context.py:164)
- [prepare_regex_context.py](/Users/nme/Documents/pythonProjects/Reviewer/prepare_regex_context.py:174)

Then it scores each citing paper with simple heuristics:

- full target title match: high score
- alias match: high score
- title phrase match: medium score
- title term match: lower score
- target vocabulary plus usage cue: extra score

Usage cues include words such as:

```text
adopt, adapt, apply, baseline, benchmark, build, compare, extend,
fine-tune, pretrain, reproduce, use, utilize
```

Relevant code:

- [prepare_regex_context.py](/Users/nme/Documents/pythonProjects/Reviewer/prepare_regex_context.py:81)
- [prepare_regex_context.py](/Users/nme/Documents/pythonProjects/Reviewer/prepare_regex_context.py:200)

## 3. How Examples Are Bucketed

After scoring, the script creates three heuristic buckets:

- candidate positive examples
- candidate uncertain examples
- candidate negative examples

Relevant code:

- [prepare_regex_context.py](/Users/nme/Documents/pythonProjects/Reviewer/prepare_regex_context.py:238)

The bucket logic is:

```text
candidate positive: score >= 5 and abstract exists
candidate uncertain: score between 1 and 4, or abstract is missing
candidate negative: score == 0 and abstract exists
```

These buckets are not final classifications. They are examples shown to the LLM
so it can design better regexes.

## 4. What Goes Into `regex_context.md`

`regex_context.md` is a Markdown evidence file. It contains:

- target paper title, DOI, and OpenAlex ID
- count of fetched citation rows
- count of rows with abstracts
- classification goal for `yes`, `no`, and `dont_know`
- aliases supplied by the user
- title-derived phrases and terms
- paper type and publication year summaries
- frequent non-target terms in the citation set
- candidate positive examples
- candidate uncertain examples
- candidate negative examples
- requested LLM output

Relevant code:

- [prepare_regex_context.py](/Users/nme/Documents/pythonProjects/Reviewer/prepare_regex_context.py:307)
- [prepare_regex_context.py](/Users/nme/Documents/pythonProjects/Reviewer/prepare_regex_context.py:332)

Each example includes:

- title
- DOI
- heuristic score and reason
- abstract snippet

Relevant code:

- [prepare_regex_context.py](/Users/nme/Documents/pythonProjects/Reviewer/prepare_regex_context.py:285)

## 5. How The LLM Turns The Context Into Regexes

The LLM does not receive only `regex_context.md`. It receives:

```text
paper_usage_regex_prompt.md
regex_context.md
```

`paper_usage_regex_prompt.md` defines the task and the desired labels:

- `yes`
- `no`
- `dont_know`

It asks the LLM to produce Python-compatible regex lists:

```python
STRONG_YES = [...]
WEAK_YES = [...]
BACKGROUND_ONLY = [...]
AMBIGUOUS = [...]
```

`llm_generate_regex.py` combines the instruction prompt and the context file,
sends them to the selected LLM provider, extracts the returned Python code
block, validates it, and writes the generated regex file.

Relevant code:

- [llm_generate_regex.py](/Users/nme/Documents/pythonProjects/Reviewer/llm_generate_regex.py:39)
- [llm_generate_regex.py](/Users/nme/Documents/pythonProjects/Reviewer/llm_generate_regex.py:64)

Example:

```bash
python3 llm_generate_regex.py \
  --provider openai \
  --model gpt-5.4-mini \
  --prompt paper_usage_regex_prompt.md \
  --context MAEex/regex_context.md \
  --output MAEex/generated_keywords_regexp.py
```

## 6. How The Generated Regexes Are Used

After regex generation, `keywords.py` classifies every row in `CitePaper.csv`:

```bash
python3 keywords.py \
  --input-csv MAEex/CitePaper.csv \
  --metadata MAEex/CitePaper.meta.json \
  --mode generated \
  --regexp MAEex/generated_keywords_regexp.py \
  --output MAEex/classified_papers.csv
```

The generated regex file is trusted local Python. It is loaded by path, compiled,
and then applied to each paper title + abstract.

The final output is:

```text
classified_papers.csv
```

with audit columns such as:

- `verdict`
- `reason`
- `matched_patterns`
- `classifier_mode`

## 7. Direct LLM Classification Without Regexes

Use this route when you want the LLM to classify papers directly instead of
first generating regular expressions.

Abstract-only classification:

```bash
python3 llm_classify_papers.py \
  --input-csv CitePaper.csv \
  --metadata CitePaper.meta.json \
  --provider openai \
  --model MODEL \
  --evidence-mode abstract \
  --output classified_papers.csv
```

Full-text classification:

```bash
python3 download_open_pdfs.py --input CitePaper.csv --out-dir open-pdfs
.venv/bin/python extract_pdf_text.py --manifest open-pdfs/manifest.csv --out-dir open-pdfs-text

python3 llm_classify_papers.py \
  --input-csv CitePaper.csv \
  --metadata CitePaper.meta.json \
  --provider openai \
  --model MODEL \
  --evidence-mode fulltext \
  --full-text-dir open-pdfs-text \
  --full-text-chars 30000 \
  --output classified_papers.csv
```

For full-text classification, PDF download and text extraction must happen
before classification. In the regex path, full text is not needed until review
prompt creation.

The LLM classifier sends one paper per API request. It asks the model to return
structured JSON with:

- `verdict`: `yes`, `no`, or `dont_know`
- `reason`: short explanation
- `evidence_quote`: short quote from the supplied abstract/full text
- `confidence`: `high`, `medium`, or `low`

If the model returns `maybe`, the script normalizes it to `dont_know`.

The output keeps the same downstream columns used by `build_review_prompt.py`
and adds LLM audit columns:

```text
openalex_id, evidence_source, evidence_quote, confidence, llm_provider, llm_model
```

This path sends abstracts and optionally extracted full-text snippets to the
selected cloud provider, so use it with the same cost and privacy caution as the
cloud review-writing scripts.
