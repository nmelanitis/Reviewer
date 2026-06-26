# Reviewer

Reviewer is a small citation-review pipeline. Given one target paper, it fetches
papers that cite it, prepares regex/classification context, classifies whether
citing papers appear to use the target paper, and builds review-writing prompts.

## Examples

Two examples demonstrate using the tool, to review all papers that cite and **use** "Masked Autoencoders Are Scalable Vision Learners" (MAE):
1. in MAEex sub-dir, we classify papers as using/not using MAE approach with a REG EXP. Run MAEex/run_mae_full_example.py 
2. in MAEllm sub-dir, we classify papers as using/not using MAE approach by prompting an LLM. Run MAEllm/run_mae_llm_fulltext_example.py

Most important output files are included in MAEex and MAEllm directories, other are ommited. 

Take a look at MAEex/Discussion.txt and MAEllm/CompareREGEx-LLM.txt files for discussion points and a comparison of the two (regex, llm) classification approaches.

CONFIG: Dont forget to set at least one cloud llm API key (see .env.example). Gemini gives a free API key, but it gets busy!

## Documentation 

Special README files explain:
1. Classify-papers-as-relevant-README.md: How we filter papers as using or not using the approach of the target paper? Different strategies are implemented in code and explained in this markdown file
2. Control-Review-size.README.md: This note explains how to control the size of prompts and outputs
3. This file shows the main Reviewer execution paths. The paths are intentionally different because regex classification uses abstracts/titles/metadata, while direct LLM full-text classification needs extracted full text before it can
 lassify papers.

## License

This project is released under the Creative Commons
Attribution-NonCommercial 4.0 International license (`CC BY-NC 4.0`).

You may use, share, and adapt it for non-commercial purposes with attribution.
Commercial use requires separate written permission.

Suggested attribution:

```text
Reviewer citation-review pipeline.
Licensed under CC BY-NC 4.0.
https://creativecommons.org/licenses/by-nc/4.0/
Please cite the repository URL when reusing or adapting this work.
```

See [LICENSE](LICENSE) for details.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pypdf PyPDF2 PyMuPDF pandas
```

`PyMuPDF` provides the `fitz` import used by `extract_pdf_text.py`.

`pdftotext` is not required. It depends on Poppler system headers and may fail
to install unless Poppler is installed on the machine.

Optional cloud LLM support:

```bash
pip install openai anthropic google-genai
```

Set the API key for the provider you want to use. You can either export keys in
your shell:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

or copy the local template and edit `.env`:

```bash
cp .env.example .env
```

Then put your real key in `.env`:

```text
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

`.env` is ignored by git. `.env.example` is safe to commit because it contains
only empty placeholders.

Important API/account distinction:

- There is no API key for "free ChatGPT" in the web app.
- Your ChatGPT Free/Plus/Pro login is not the same thing as an API key.
- OpenAI API usage is managed through the OpenAI Platform and billed separately
  from ChatGPT subscriptions.
- Create/manage OpenAI API keys at `https://platform.openai.com/api-keys`.
- The key represents your OpenAI Platform project/account for API calls. Treat
  it like a password.

Cloud LLM calls may cost money. The Markdown prompt content is sent to the
selected provider.

## Step 1: Fetch Citing Papers

```bash
python3 fetch_openalex_citations.py 10.48550/arxiv.2111.06377
```

Outputs:

- `CitePaper.csv`
- `CitePaper.meta.json`

Useful smoke-test option:

```bash
python3 fetch_openalex_citations.py 10.48550/arxiv.2111.06377 --max-results 5
```

`CitePaper.csv` includes title, abstract, DOI, OpenAlex IDs, open-access status,
and OpenAlex-provided PDF URLs when available.

## Step 2: Prepare Regex Context For LLM

```bash
python3 prepare_regex_context.py --alias MAE --alias "masked autoencoder"
```

Outputs:

- `regex_context.md`

To use an LLM for regex generation:

1. Open `paper_usage_regex_prompt.md`.
2. Open `regex_context.md`.
3. Paste both into ChatGPT, Claude, or a local model.
4. Ask it to produce Python-compatible regex lists.
5. Put the regex lists into a Python file like `example_generated_keywords_regexp.py`.

The generated regex file should define:

```python
STRONG_YES = [...]
WEAK_YES = [...]
BACKGROUND_ONLY = [...]
AMBIGUOUS = [...]
```

### How Papers Are Classified As Yes / Maybe / No: Prompt Creation And LLM Regex Generation

The generated-regex path has two prompt-building pieces:

1. `prepare_regex_context.py` creates `regex_context.md`.
2. `llm_generate_regex.py` sends `paper_usage_regex_prompt.md` plus
   `regex_context.md` to the LLM and writes a Python regex file.

`prepare_regex_context.py` reads the unclassified citation set and target
metadata. It parses:

> Important: the regular expression is built using information from the
> abstracts of the citing papers, together with titles, target metadata, and
> user-provided aliases.

- Target metadata from `CitePaper.meta.json`: target title, DOI, and OpenAlex ID.
- Citing-paper rows from `CitePaper.csv`: title / `display_name`, abstract, DOI,
  paper type, publication year, and cited-by count.
- Optional user aliases such as `MAE`, `masked autoencoder`, or other
  target-specific names.

The script tokenizes the target title, removes stopwords, and derives title
terms and title phrases. It then scores each citing paper using simple
heuristics: full target title match, alias match, title phrase match, title term
match, and target vocabulary plus usage cues such as `use`, `adapt`, `apply`,
`benchmark`, or `extend`.

The output `regex_context.md` contains target metadata, vocabulary seeds,
summary statistics, and capped examples grouped as candidate positive,
uncertain, and negative cases. These are not final labels; they are evidence for
the LLM to design regexes. Example count and abstract snippet length are
controlled with:

```bash
python3 prepare_regex_context.py \
  --max-examples 12 \
  --snippet-chars 2500 \
  --alias MAE \
  --alias "masked autoencoder"
```

For a fuller explanation, see `Classify-papers-as-relevant-README.md`.

## Step 3A: Classify Without LLM Regex

This uses metadata-derived regexes from the target title plus optional aliases.

```bash
python3 keywords.py --alias MAE --alias "masked autoencoder"
```

Output:

- `classified_papers.csv`

This is the default no-LLM path.

## Step 3B: Classify With Generated Regex

Use this after pasting LLM-generated regexes into a Python file.

```bash
python3 keywords.py --mode generated --regexp example_generated_keywords_regexp.py
```

Output:

- `classified_papers.csv`

The generated regex file is trusted local Python.

## Step 3C: Generate Regex With A Cloud LLM

First prepare the regex context:

```bash
python3 prepare_regex_context.py --alias MAE --alias "masked autoencoder"
```

Then ask a cloud LLM to produce a regex file:

```bash
python3 llm_generate_regex.py \
  --provider openai \
  --model MODEL \
  --prompt paper_usage_regex_prompt.md \
  --context regex_context.md \
  --output generated_keywords_regexp.py
```

Other providers:

```bash
python3 llm_generate_regex.py --provider anthropic --model MODEL --output generated_keywords_regexp.py
python3 llm_generate_regex.py --provider gemini --model MODEL --output generated_keywords_regexp.py
```

Then classify with that generated file:

```bash
python3 keywords.py --mode generated --regexp generated_keywords_regexp.py
```

The LLM-generated file is validated before it is written. It must define:

```python
STRONG_YES = [...]
WEAK_YES = [...]
BACKGROUND_ONLY = [...]
AMBIGUOUS = [...]
```

## Step 3D: Classify Directly With A Cloud LLM

This path skips regex creation entirely. Instead, it sends one citing paper at a
time to a cloud LLM and asks for a structured `yes`, `no`, or `dont_know`
classification. If the model says "maybe", the script normalizes that to
`dont_know`.

Abstract-only classification:

```bash
python3 llm_classify_papers.py \
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
  --provider openai \
  --model MODEL \
  --evidence-mode fulltext \
  --full-text-dir open-pdfs-text \
  --full-text-chars 30000 \
  --output classified_papers.csv
```

Full-text mode uses extracted text when available. If a paper has no extracted
full text but has an abstract, the classifier falls back to the abstract and
records `evidence_source=abstract_fallback`.

Useful smoke-test and audit options:

```bash
python3 llm_classify_papers.py \
  --provider gemini \
  --model MODEL \
  --evidence-mode abstract \
  --limit 2 \
  --raw-dir llm-classification-raw
```

The output keeps the normal downstream columns:

```text
display_name, doi, abstract, verdict, reason, matched_patterns, classifier_mode
```

and adds LLM audit columns:

```text
openalex_id, evidence_source, evidence_quote, confidence, llm_provider, llm_model
```

Cloud classification sends title, DOI, abstract, and optionally extracted full
text snippets to the selected provider. This can cost money and may send paper
content outside your machine.

## Step 4A: Abstract-Only Review Prompt

```bash
python3 build_review_prompt.py
```

Output:

- `review_prompt.md`

This prompt is based on abstracts, metadata, and classifier audit fields. It
tells the LLM to use cautious language because full papers may not have been
reviewed.

By default, `build_review_prompt.py` includes only papers classified as `yes`.
This is intentional: the review is about how the target paper/model was used in
the literature.

To include uncertain papers:

```bash
python3 build_review_prompt.py --include-dont-know
```

To include a small sample of papers classified as `no`:

```bash
python3 build_review_prompt.py --include-no
```

Control how much text goes into the Markdown prompt:

```bash
python3 build_review_prompt.py --abstract-chars 900
python3 build_review_prompt.py --abstract-chars all
```

Important: the LLM does not see `classified_papers.csv` directly. It sees the
generated Markdown file, `review_prompt.md`. If `--abstract-chars 900` is used,
the LLM sees at most 900 abstract characters per listed paper. If
`--abstract-chars all` is used, the LLM sees the full abstract for each listed
paper.

## Step 4B: Full-Text Route

Download open PDFs:

```bash
python3 download_open_pdfs.py \
  --input CitePaper.csv \
  --out-dir open-pdfs \
  --max-pdfs 10
```

`--input` is the citation CSV to read. It must contain the PDF/open-access URL
columns written by `fetch_openalex_citations.py`. `--out-dir` is the folder
where PDFs and `manifest.csv` are saved.

Extract text:

```bash
.venv/bin/python extract_pdf_text.py \
  --manifest open-pdfs/manifest.csv \
  --out-dir open-pdfs-text
```

`--manifest` tells the extractor which downloaded PDFs to read. `--out-dir` is
the folder where `.txt` files and the extraction `manifest.csv` are saved.

Build a prompt that includes extracted full-text snippets when available:

```bash
python3 build_review_prompt.py --full-text-dir open-pdfs-text --output review_prompt_fulltext.md
```

Control how much extracted full text goes into the Markdown prompt:

```bash
python3 build_review_prompt.py \
  --full-text-dir open-pdfs-text \
  --full-text-chars 2000 \
  --output review_prompt_fulltext.md

python3 build_review_prompt.py \
  --full-text-dir open-pdfs-text \
  --full-text-chars all \
  --output review_prompt_fulltext.md
```

Important: the LLM does not receive PDF files. It receives the generated
Markdown file. If `--full-text-chars 2000` is used, the LLM sees at most 2000
characters of extracted text per listed paper. If `--full-text-chars all` is
used, the LLM sees all extracted text included for each listed paper. Whole full
texts can easily exceed a model context window, so use `all` carefully.

Outputs:

- `open-pdfs/`
- `open-pdfs/manifest.csv`
- `open-pdfs-text/`
- `open-pdfs-text/manifest.csv`
- `review_prompt_fulltext.md`

## Step 5: Orchestrator

Run the default no-LLM, abstract-only workflow:

```bash
python3 run_pipeline.py 10.48550/arxiv.2111.06377 --alias MAE --alias "masked autoencoder"
```

Prepare regex context for manual LLM regex generation:

```bash
python3 run_pipeline.py 10.48550/arxiv.2111.06377 \
  --alias MAE \
  --alias "masked autoencoder" \
  --with-regex-context
```

Run with a generated regex file:

```bash
python3 run_pipeline.py 10.48550/arxiv.2111.06377 \
  --mode generated \
  --regexp example_generated_keywords_regexp.py
```

Run with open PDF download and text extraction:

```bash
python3 run_pipeline.py 10.48550/arxiv.2111.06377 \
  --alias MAE \
  --citation-csv CitePaper.csv \
  --with-pdfs \
  --pdf-dir open-pdfs \
  --max-pdfs 10 \
  --with-fulltext \
  --fulltext-dir open-pdfs-text
```

Run with direct LLM abstract classification:

```bash
python3 run_pipeline.py 10.48550/arxiv.2111.06377 \
  --classifier llm \
  --llm-provider openai \
  --llm-model MODEL \
  --llm-classify-evidence abstract
```

Run with direct LLM full-text classification:

```bash
python3 run_pipeline.py 10.48550/arxiv.2111.06377 \
  --classifier llm \
  --llm-provider openai \
  --llm-model MODEL \
  --llm-classify-evidence fulltext \
  --with-pdfs \
  --pdf-dir open-pdfs \
  --with-fulltext \
  --fulltext-dir open-pdfs-text \
  --llm-classify-full-text-chars 30000
```

For direct LLM full-text classification, the orchestrator downloads PDFs and
extracts text before classification. This is different from the regex path,
where PDF download/extraction can happen after classification for review-prompt
building.

Smoke-test with fewer OpenAlex rows:

```bash
python3 run_pipeline.py 10.48550/arxiv.2111.06377 --alias MAE --max-results 5
```

The orchestrator calls the existing scripts in order. By default it does not
call an LLM. Cloud LLM steps run only when explicit `--llm-*` flags are used.

For full-text extraction, `run_pipeline.py` uses `.venv/bin/python` automatically
when it exists because the PDF extraction libraries are installed there. You can
override this with `--extract-python`.

Path options for the full-text route:

- `--citation-csv`: CSV fetched from OpenAlex and used by the PDF downloader.
- `--pdf-dir`: folder where PDFs and the PDF download manifest are stored.
- `--fulltext-dir`: folder where extracted `.txt` files and extraction manifest are stored.

## Step 5B: Write The Final Review With A Cloud LLM

First build a review prompt. Abstract-only:

```bash
python3 build_review_prompt.py --output review_prompt.md
```

Full-text-aware:

```bash
python3 build_review_prompt.py \
  --full-text-dir open-pdfs-text \
  --full-text-chars 2000 \
  --output review_prompt_fulltext.md
```

Then send the Markdown prompt to a cloud LLM:

```bash
python3 llm_write_review.py \
  --provider openai \
  --model MODEL \
  --mode selective \
  --prompt review_prompt.md \
  --output review_draft.md
```

For full-text prompts:

```bash
python3 llm_write_review.py \
  --provider openai \
  --model MODEL \
  --mode selective \
  --prompt review_prompt_fulltext.md \
  --output review_draft_fulltext.md
```

Review-writing modes:

- `single`: sends the whole Markdown prompt in one request. Good for small
  abstract-only prompts.
- `selective`: removes omitted-group bookkeeping and unavailable full-text lines
  before sending. This is the default and best first choice for cost/context
  control.
- `chunked`: splits a large prompt into chunks, asks the LLM to summarize each
  chunk, then asks for a final synthesis from chunk summaries.

Chunked example:

```bash
python3 llm_write_review.py \
  --provider openai \
  --model MODEL \
  --mode chunked \
  --prompt review_prompt_fulltext.md \
  --chunk-chars 60000 \
  --chunks-dir review_chunks \
  --output review_draft_fulltext.md
```

Context control:

- `build_review_prompt.py --max-yes` controls how many `yes` papers are listed.
- `build_review_prompt.py --include-dont-know` opts in uncertain papers.
- `build_review_prompt.py --include-no` opts in a no-use/background sample.
- `build_review_prompt.py --abstract-chars` controls abstract characters per paper.
- `build_review_prompt.py --full-text-chars` controls extracted full-text characters per paper.
- `llm_write_review.py --max-input-chars` prevents sending prompts that are too large.
- `llm_write_review.py --mode chunked --chunk-chars 60000` handles larger prompts by map-reduce summarization.

## Write The LLM Review: Input And Output Limits

`llm_write_review.py` has two main size controls:

- `--max-input-chars`: local safety limit for the Markdown prompt before it is
  sent to the provider. This is measured in characters, not exact tokens.
- `--max-output-tokens`: maximum response tokens requested from the provider.

The script does not currently count exact input tokens. As a rough estimate for
English academic text:

```text
4 characters ~= 1 token
```

So these are approximately equivalent:

```text
120000 characters ~= 30000 tokens
800000 characters ~= 200000 tokens
3000000 characters ~= 750000 tokens
```

Default behavior:

```bash
python3 llm_write_review.py \
  --provider gemini \
  --model gemini-2.5-flash-lite \
  --prompt review_prompt.md \
  --output review_draft.md
```

uses:

```text
--max-input-chars 120000
--max-output-tokens 4096
```

Small smoke test:

```bash
python3 llm_write_review.py \
  --provider gemini \
  --model gemini-2.5-flash-lite \
  --mode selective \
  --prompt review_prompt.md \
  --output review_draft_smoke.md \
  --max-output-tokens 512
```

Large GPT-5.5-style single-shot run:

```bash
python3 llm_write_review.py \
  --provider openai \
  --model gpt-5.5 \
  --mode single \
  --prompt review_prompt_fulltext.md \
  --output review_draft_fulltext.md \
  --max-input-chars 3000000 \
  --max-output-tokens 128000
```

That example is close to "maximum" for a model with a `1M` token context window
and a `128K` token output limit, because `3,000,000` input characters is roughly
`750K` input tokens. It leaves room for instructions, tokenization variance, and
the generated answer. Do not set `--max-input-chars` to the full theoretical
context converted to characters unless you are willing to have the provider
reject the request.

For very large prompts, prefer chunked mode:

```bash
python3 llm_write_review.py \
  --provider openai \
  --model gpt-5.5 \
  --mode chunked \
  --prompt review_prompt_fulltext.md \
  --chunks-dir review_chunks \
  --chunk-chars 60000 \
  --max-input-chars 3000000 \
  --max-output-tokens 128000 \
  --output review_draft_fulltext.md
```

In chunked mode, `--chunk-chars` controls the size of each evidence chunk sent
for summarization. `--max-input-chars` still protects each individual request,
including the final synthesis prompt.

## Write The LLM Review: How Many Abstracts Or Full Texts?

For `gpt-5.5`, OpenAI's model docs list a `1M` token context window and a
`128K` token maximum output. That is a large window, but the practical budget is
smaller because the prompt also contains instructions, target metadata, titles,
DOIs, classifier reasons, matched regexes, and space for the model to reason and
write the review.

The pipeline uses character caps because they are easy to inspect before sending
data to a cloud provider. A rough conversion for English academic text is:

```text
4 characters ~= 1 token
```

Current project estimate for `10.48550/arxiv.2111.06377`:

- Default `review_prompt.md`: about `18.8K` characters, or about `4.7K` tokens.
- `yes` abstracts only: `13` papers, about `19.4K` abstract characters.
- All available abstracts: `145` abstracts, about `192K` abstract characters,
  or about `48K` tokens before prompt overhead.
- Current extracted full text: `1` paper, about `147K` characters, or about
  `37K` tokens.

This means abstract-only review is easy for `gpt-5.5`, even if you include all
available abstracts. Full text grows much faster. If future extracted full texts
look like the current one, a single prompt can likely hold around `15-20` whole
papers comfortably, but chunked mode is safer once you move beyond a small set.

Good defaults:

```bash
python3 build_review_prompt.py \
  --full-text-dir open-pdfs-text \
  --full-text-chars 8000 \
  --output review_prompt_fulltext.md
```

Then write with selective mode:

```bash
python3 llm_write_review.py \
  --provider openai \
  --model gpt-5.5 \
  --mode selective \
  --prompt review_prompt_fulltext.md \
  --output review_draft_fulltext.md
```

To send all abstracts:

```bash
python3 build_review_prompt.py \
  --abstract-chars all \
  --output review_prompt.md
```

To send whole extracted full texts:

```bash
python3 build_review_prompt.py \
  --full-text-dir open-pdfs-text \
  --full-text-chars all \
  --output review_prompt_fulltext.md
```

If the whole-text prompt is too large for the default safety cap, either raise
the cap:

```bash
python3 llm_write_review.py \
  --provider openai \
  --model gpt-5.5 \
  --mode single \
  --prompt review_prompt_fulltext.md \
  --max-input-chars 800000 \
  --output review_draft_fulltext.md
```

or use chunked mode:

```bash
python3 llm_write_review.py \
  --provider openai \
  --model gpt-5.5 \
  --mode chunked \
  --prompt review_prompt_fulltext.md \
  --chunk-chars 60000 \
  --output review_draft_fulltext.md
```

Recommended rule of thumb:

- Abstract-only: include all abstracts if useful.
- Full-text snippets: start with `--full-text-chars 8000` or `12000`.
- Whole full texts: use only for a small set of `yes` papers, or switch to
  `--mode chunked`.
- Keep single-shot prompts below roughly `600K-750K` tokens when possible, even
  if the model's hard context window is larger.

Run the orchestrator with cloud regex generation:

```bash
python3 run_pipeline.py 10.48550/arxiv.2111.06377 \
  --alias MAE \
  --llm-provider openai \
  --llm-model MODEL \
  --llm-generate-regex \
  --llm-regex-output generated_keywords_regexp.py
```

Run the orchestrator with cloud review writing:

```bash
python3 run_pipeline.py 10.48550/arxiv.2111.06377 \
  --alias MAE \
  --llm-provider openai \
  --llm-model MODEL \
  --llm-write-review \
  --llm-review-mode selective \
  --llm-review-output review_draft.md
```

Run the orchestrator with full texts and cloud review writing:

```bash
python3 run_pipeline.py 10.48550/arxiv.2111.06377 \
  --alias MAE \
  --with-pdfs \
  --max-pdfs 10 \
  --with-fulltext \
  --fulltext-dir open-pdfs-text \
  --llm-provider openai \
  --llm-model MODEL \
  --llm-write-review \
  --llm-review-mode chunked \
  --llm-review-output review_draft_fulltext.md
```
