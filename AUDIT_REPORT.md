# Audit of the Arabic LLM study: code, results and paper

Scope: the complete run archive of the study — the scripts as executed
(`original_as_run/`), the result JSONs and logs they produced (`original_results/`), and
the prepared datasets — checked against the manuscript draft.

Every number below can be regenerated on CPU, without downloading a single model:

```bash
python audit/audit_track_b.py --root <path to the run archive> --out results_audit
```

The archive referred to below as `results/...` is the original working tree; its scripts
and outputs are preserved verbatim in `original_as_run/` and `original_results/`. The
corrected pipeline that resolves every defect listed here is in `pipeline/`.

## 1. What the run archive actually contains

| Item | Status |
|---|---|
| `results/03_track_b_evaluate.py` (modified 30 May) | The code that produced the **Jais** (21–22 Apr) and **Falcon** (30 May) results. |
| ALLaM (4 Apr) and Qwen (6 Apr) results | Produced by an **earlier revision** of that script, which was **not preserved**. |
| `run_falcon_only.py` + `logs/falcon_only_rerun_stdout.log` | The Falcon rerun. It is the **only** run with logged raw generations: the first 5 items per dataset, 30 in total. |
| `outputs/falcon/falcon_raw_outputs.jsonl` (all 17,708 Falcon generations) | **Referenced, but not present in the archive.** |
| `prepared_data/` | Created once on 1 Apr and never modified, so every run used identical data. |
| `chunking_eval/` | Unrelated RAG chunking project; ignored. |

The paper's Appendix A extractor and Appendix B transcripts were never produced by this
code. The PDF's own footnotes mark them as placeholders. No "final version" containing
an EF (extraction-failure) state is missing: none of the surviving revisions has one.

## 2. Track B: every cell audited

"Best constant" is the accuracy of always answering the single most frequent gold letter.

| Model | Dataset | Reported | Chance | Best constant | Status |
|---|---|---:|---:|---:|---|
| ALLaM-7B | Arabic MMLU | 0.0 | 29.3 | A 31.2 | Gold-label defect (zero anomaly) |
| ALLaM-7B | EXAMS AR | **53.4** | 25.0 | B 26.3 | **Valid** (p = 1e-40) |
| ALLaM-7B | Madinah QA | 0.0 | 25.3 | A 26.7 | Gold-label defect (zero anomaly) |
| ALLaM-7B | ArabCulture | 0.0 | 33.3 | A 34.1 | Gold-label defect (zero anomaly); stem missing too |
| ALLaM-7B | AraDiCE | 0.0 | 33.3 | – | Gold missing |
| ALLaM-7B | AraTrust | 25.1 | 33.3 | C 36.6 | Prompt defect (no options shown) |
| Falcon-7B | Arabic MMLU | 29.6 | 29.3 | A 31.2 | Degenerate output, at or below always-A |
| Falcon-7B | EXAMS AR | 21.0 | 25.0 | B 26.3 | Degenerate output |
| Falcon-7B | Madinah QA | 26.1 | 25.3 | A 26.7 | Degenerate output |
| Falcon-7B | ArabCulture | 33.7 | 33.3 | A 34.1 | Degenerate output; stem missing |
| Falcon-7B | AraDiCE | 0.0 | 33.3 | – | Gold missing |
| Falcon-7B | AraTrust | 26.6 | 33.3 | C 36.6 | Prompt defect |
| Jais-13B | Arabic MMLU | 26.6 | 29.3 | A 31.2 | **3,846 correct = exactly the number of gold-B items** |
| Jais-13B | EXAMS AR | 0.2 | 25.0 | B 26.3 | Degenerate (1-token budget) |
| Jais-13B | Madinah QA | n/a | – | – | No result file |
| Jais-13B | ArabCulture | 33.5 | 33.3 | A 34.1 | ≈ always-B (367 vs 371); stem missing |
| Jais-13B | AraDiCE | 0.0 | 33.3 | – | Gold missing |
| Jais-13B | AraTrust | 32.2 | 33.3 | C 36.6 | Prompt defect; **168 = exactly the number of gold-B items** |
| Qwen-7B | Arabic MMLU | 0.0 | 29.3 | A 31.2 | Gold-label defect (zero anomaly) |
| Qwen-7B | EXAMS AR | **49.0** | 25.0 | B 26.3 | **Valid** (p = 2e-29) |
| Qwen-7B | Madinah QA | 0.0 | 25.3 | A 26.7 | Gold-label defect (zero anomaly) |
| Qwen-7B | ArabCulture | 0.0 | 33.3 | A 34.1 | Gold-label defect (zero anomaly) |
| Qwen-7B | AraDiCE | 0.0 | 33.3 | – | Gold missing |
| Qwen-7B | AraTrust | 23.8 | 33.3 | C 36.6 | Prompt defect |

Of the 23 completed runs, **2 are capability measurements**. The other 21 come from
harness defects. Full per-cell reasons are in `results_audit/track_b_audit.csv`.

## 3. The defects

**D1. AraDiCE has no gold labels.** The prepared columns
are `QID, Question, Option A, Option B, Option C`, and the HF dataset ships no answer
column. The official lm-evaluation-harness task sets `doc_to_target: 0`, meaning Option A
is always the correct answer (checked at commit d6de816, `lm_eval/tasks/aradice/cultural-benchmark/*.yaml`).
Our `get_gold_answer()` returns `""` for all 120 items. The fix is gold = A plus a
deterministic option shuffle; without the shuffle, a model that always says "A" would
score 100%.

**D2. The ALLaM and Qwen zeros are a gold-lookup problem, not extraction** (new finding).
Both models score exactly 0 on the three datasets whose gold field is `Answer Key` or the
dict `answer_key`, and above 0 on the two whose field is `answer` or `Answer`.
- Under the surviving code, gold resolves for 100% of those items.
- The surviving extractor maps almost any Arabic text to a letter, so a real model can't hit 0/14,455.
- Jais and Falcon, run later with the surviving code, score above 0 on the same three datasets.

The most likely cause is that the unpreserved April revision did not look up those field
names. This is inferred, not verified; the verification step is in §6.

**D3. Prompts dropped content.**
- `build_trust_prompt()` never shows AraTrust's options (522 of 522 prompts).
- ArabCulture's scenario sentence is in `first_statement`, which the builder never reads (1,097 of 1,097 prompts).
- The `Context` passage is dropped for 602 of 977 Madinah QA items and 706 of 14,455 Arabic MMLU items.

Also:
- The instruction always says "(أ، ب، ج، أو د)", even for 2- or 5-option items.
- AraTrust has 3 options, not 4, so the paper's "25% four-way chance" is wrong.

**D4. Falcon's output is degenerate repetition.** All 30 logged Falcon
generations are one character repeated to the token limit; 24 of them are
`ألللللللللللللللللل`. Rule 2 of `extract_answer()` maps anything that starts with alef
to "A" (25 of 30). Falcon's score is at or below always-A on every dataset. A one-item
control without batching also produced a malformed single-byte output
(`logs/falcon_dynamic_cache_error_before_fix.log`), so batching alone isn't the explanation. The root
cause was not isolated.

**D5. Jais was run with `max_new_tokens=1`**, while the paper says 20. Its correct counts
equal the gold-B counts exactly on MMLU (3,846) and AraTrust (168).

**D6. The extractor has no EF state.** It always returns something:
- a letter taken from the first character, then from anywhere in the text;
- otherwise `text[:10]`, which is silently scored as wrong.

Its failure mode is **false positives**: garbage becomes a letter.

**Track A.**
- (a) `load_best_model_at_end` used the **test** split as `eval_dataset`, so the reported checkpoint is the best-on-test of 3 epochs. The effect is mild but must be disclosed.
- (b) ANERcorp was run with `task: classification` and one **word per input**: context-free BIO-tag classification with macro-F1 over 9 tags, not NER. `run_ner()` exists but was never called.
- The Track A numbers themselves match the paper exactly.

## 4. How the paper's claims change

| Paper claim | Status |
|---|---|
| Zeros are extraction failures of verbose output (Abstract, contribution 2, VI-C, IX, X) | **Wrong mechanism.** The zeros come from gold-label defects. The broader claim, that the harness rather than the model sets the score and leaderboards rank harness behaviour, holds and is stronger. |
| Falcon 21–34% is "genuine, parseable performance" | **Wrong.** It's degenerate output at or below the always-A baseline. |
| AraDiCE 0% comes from dialectal preambles (App. B, Case 2) | **Fictional.** There is no gold column. |
| Appendix A 5-step extractor with EF | **Does not match the code.** Replaced with the real procedure. |
| Appendix B ALLaM transcript | **Fictional.** No ALLaM raw output survives. Replaced with the real Falcon logs. |
| 20-token budget for all models | Wrong for Jais (1 token). |
| AraTrust near the 25% four-way chance | AraTrust is 3-way (33.3%) and the options were not shown. |
| Fig. 3 "parseable on k of 6" | Meaningless under this extractor. Replaced by a score-vs-baseline figure. |
| Jais Madinah "exhausted host memory" | **No log supports this.** Revised text says only that no result file was produced. Keep your explanation only if you have evidence. |
| Track A "boundary-sensitive NER" | Reworded to word-level tagging, with the checkpoint-selection disclosure added. |

## 5. What is delivered

- `audit/audit_track_b.py`: the audit above (CPU only).
- `pipeline/`: the corrected code.
  - `harness.py`: explicit gold schemas that raise on a missing gold label, complete prompts, a strict extractor with an UNPARSED state, AraDiCE shuffling, and constant-letter baselines.
  - `03_track_b_evaluate.py`: a log-likelihood mode (default) and a generate mode; the same 20-token budget for every model; batch size 1; chat templates; raw JSONL logging for every item; parse rates. `--harness legacy` reproduces the as-run behaviour.
  - `02_track_a_finetune.py`: validation-based checkpoint selection.
  - `tests/`: 41 CPU tests against the real prepared data, all passing. They include a sanity gate: an always-A predictor must score exactly its baseline and never 0.
  - `07_make_paper_figures.py`: regenerates the new figures.
- `paper/revised_sections.tex`: drop-in LaTeX for every affected passage. `paper/preview_revised_sections.pdf` is a preview build.
- `original_as_run/` and `original_results/`: untouched provenance copies.

## 6. Recommended reruns on the RTX 4060 (optional, strengthens the paper)

1. **Confirm D2 (about 15 min):**
   `python 03_track_b_evaluate.py --model allam --dataset arabic_mmlu --mode generate --limit 300 --no-chat-template`.
   An accuracy well above 0 with a high parse rate confirms the gold-lookup explanation.
2. **Full clean Track B (primary protocol):** `python 03_track_b_evaluate.py --all --mode loglik`. This replaces Fig. 2 with real numbers. Jais is slow because of the CPU offload.
3. **Track A with validation selection:** `python 02_track_a_finetune.py --all`. This removes the disclosure caveat.
