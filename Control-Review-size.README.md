# Control Review Size

This note explains how to control the size of prompts and outputs when using
`llm_write_review.py`.

There are two different size controls:

- `--max-input-chars`: local safety cap for the Markdown prompt before it is
  sent to the LLM.
- `--max-output-tokens`: requested maximum length of the LLM answer.

Input size is controlled in characters, not exact tokens. Output size is
controlled in tokens, because that is what provider APIs expose.

## 1. Limit The Input Prompt Size

This prevents accidentally sending a huge prompt to a cloud provider:

```bash
python3 llm_write_review.py \
  --provider gemini \
  --model gemini-2.5-flash-lite \
  --mode single \
  --prompt review_prompt_all_abstracts.md \
  --output review_draft.md \
  --max-input-chars 200000
```

If `review_prompt_all_abstracts.md` is larger than `200000` characters, the
script stops before calling the API.

## 2. Let The Input Be As Long As The Abstracts You Chose

First build the prompt with the abstracts you want.

For all full abstracts from `yes` papers:

```bash
python3 build_review_prompt.py \
  --abstract-chars all \
  --max-yes 999999 \
  --output review_prompt_all_yes_abstracts.md
```

For all full abstracts from all verdict groups:

```bash
python3 build_review_prompt.py \
  --abstract-chars all \
  --max-yes 999999 \
  --include-dont-know \
  --max-dont-know 999999 \
  --include-no \
  --max-no 999999 \
  --output review_prompt_all_abstracts.md
```

Then set `--max-input-chars` high enough that it will not block the prompt:

```bash
python3 llm_write_review.py \
  --provider gemini \
  --model gemini-2.5-flash-lite \
  --mode single \
  --prompt review_prompt_all_abstracts.md \
  --output review_draft_all_abstracts.md \
  --max-input-chars 1000000
```

This is effectively uncapped for the current abstract-only project, because all
available abstracts are much smaller than `1000000` characters.

There is no true `--max-input-chars unlimited` option. To let the prompt be as
long as the abstracts you selected, set `--max-input-chars` comfortably above
the Markdown file size.

You can check prompt size with:

```bash
wc -c review_prompt_all_abstracts.md
```

## 3. Limit The Output Length

For a short smoke test:

```bash
python3 llm_write_review.py \
  --provider gemini \
  --model gemini-2.5-flash-lite \
  --mode selective \
  --prompt review_prompt.md \
  --output review_draft_smoke.md \
  --max-output-tokens 512
```

For a longer review draft:

```bash
python3 llm_write_review.py \
  --provider gemini \
  --model gemini-2.5-flash-lite \
  --mode selective \
  --prompt review_prompt.md \
  --output review_draft.md \
  --max-output-tokens 8192
```

## 4. Let The Output Be As Long As The Model Allows

There is no true uncapped output mode in cloud LLM APIs. Providers require or
enforce a maximum output token limit.

To approximate uncapped output, set `--max-output-tokens` near the model's
maximum supported output.

Example for a model that supports very large outputs:

```bash
python3 llm_write_review.py \
  --provider openai \
  --model gpt-5.5 \
  --mode single \
  --prompt review_prompt_all_abstracts.md \
  --output review_draft_long.md \
  --max-input-chars 1000000 \
  --max-output-tokens 128000
```

Use smaller values for smaller or cheaper models. If the provider rejects the
request, reduce `--max-output-tokens`.

## Practical Recipe For This Project

All available abstracts, full length, one API call:

```bash
python3 build_review_prompt.py \
  --abstract-chars all \
  --max-yes 999999 \
  --include-dont-know \
  --max-dont-know 999999 \
  --include-no \
  --max-no 999999 \
  --output review_prompt_all_abstracts.md

python3 llm_write_review.py \
  --provider gemini \
  --model gemini-2.5-flash-lite \
  --mode single \
  --prompt review_prompt_all_abstracts.md \
  --output review_draft_all_abstracts.md \
  --max-input-chars 1000000 \
  --max-output-tokens 8192
```

All `yes` abstracts only, full length, recommended for the actual review:

```bash
python3 build_review_prompt.py \
  --abstract-chars all \
  --max-yes 999999 \
  --output review_prompt_all_yes_abstracts.md

python3 llm_write_review.py \
  --provider gemini \
  --model gemini-2.5-flash-lite \
  --mode selective \
  --prompt review_prompt_all_yes_abstracts.md \
  --output review_draft_all_yes_abstracts.md \
  --max-input-chars 1000000 \
  --max-output-tokens 8192
```

