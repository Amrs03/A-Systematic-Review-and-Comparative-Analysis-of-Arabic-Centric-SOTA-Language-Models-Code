> [!WARNING]
> **Unaudited provenance snapshot — preserved verbatim, not a result of this study.**
>
> This file is the compiled output of the original run, kept unchanged so the audit can be
> checked against it. The leaderboard below is precisely the ranking that
> [`AUDIT_REPORT.md`](../AUDIT_REPORT.md) invalidates: 21 of the 23 Track B cells that feed
> these averages are harness artifacts, not capability measurements. Do not cite these
> numbers. Image links point at files that were not preserved.

# Arabic NLP Evaluation Findings

Generated: 2026-06-01

## Scope

- Tracks included: Track A and Track B
- Models included: ALLAM, AraBERT, Falcon, Jais, MARBERT, Qwen
- Source artifacts: existing compiled result files under `results/`
- Evaluation reruns performed for this consolidated file: none
- Latest substantive rerun included in the results: Falcon-Arabic Track B cache-fix rerun

## Data Sources

- `results/compiled_results.csv`
- `results/compiled_category_scores.csv`
- `results/compiled_leaderboard.csv`
- `results/compiled_missing_runs.csv`
- `results/compiled_report.md`
- Falcon-only artifacts under `outputs/falcon/`

## Snapshot

| Item | Value |
|---|---:|
| Result files read | 33 |
| Valid scored runs | 33 |
| Invalid / missing-score runs | 0 |
| Models with valid ranked scores | 6 |
| Missing expected result files | 1 |

## Overall Leaderboard

Scores are averaged across categories where each model has valid completed results. Track A and Track B evaluate different model families and task types, so category-level and dataset-level tables are the fairest comparisons.

| Rank | Model | Average Score | Categories Completed |
|---:|---|---:|---:|
| 1 | MARBERT | 68.2126% | 4 |
| 2 | AraBERT | 66.7766% | 4 |
| 3 | Falcon | 23.0271% | 3 |
| 4 | Jais | 20.7693% | 3 |
| 5 | ALLAM | 14.3036% | 3 |
| 6 | Qwen | 13.3600% | 3 |

## Category Findings

| Category | Model | Average Score | Datasets |
|---|---|---:|---|
| Commonsense / STEM | Falcon | 25.5887% | Arabic MMLU, Madinah QA, EXAMS AR |
| Commonsense / STEM | ALLAM | 17.8150% | Arabic MMLU, EXAMS AR, Madinah QA |
| Commonsense / STEM | Qwen | 16.3253% | Arabic MMLU, EXAMS AR, Madinah QA |
| Commonsense / STEM | Jais | 13.3965% | Arabic MMLU, EXAMS AR |
| Cultural Understanding | Falcon | 16.8642% | ArabCulture, AraDiCE |
| Cultural Understanding | Jais | 16.7274% | AraDiCE, ArabCulture |
| Cultural Understanding | ALLAM | 0.0000% | AraDiCE, ArabCulture |
| Cultural Understanding | Qwen | 0.0000% | AraDiCE, ArabCulture |
| Dialect Identification | MARBERT | 74.0440% | QADI |
| Dialect Identification | AraBERT | 68.6334% | QADI |
| Named Entity Recognition | AraBERT | 54.7147% | ANERcorp |
| Named Entity Recognition | MARBERT | 52.9878% | ANERcorp |
| Sentiment Analysis | MARBERT | 69.2034% | ASTD, HARD |
| Sentiment Analysis | AraBERT | 68.8412% | ASTD, HARD |
| Social Meaning | MARBERT | 76.6152% | ArSarcasm |
| Social Meaning | AraBERT | 74.9172% | ArSarcasm |
| Social Meaning | Jais | 32.1839% | AraTrust |
| Social Meaning | Falcon | 26.6284% | AraTrust |
| Social Meaning | ALLAM | 25.0958% | AraTrust |
| Social Meaning | Qwen | 23.7548% | AraTrust |

## Track A Dataset Results

| Dataset | Category | Model | Metric | Score |
|---|---|---|---|---:|
| QADI | Dialect Identification | MARBERT | F1 macro | 74.0440% |
| QADI | Dialect Identification | AraBERT | F1 macro | 68.6334% |
| ANERcorp | Named Entity Recognition | AraBERT | F1 macro | 54.7147% |
| ANERcorp | Named Entity Recognition | MARBERT | F1 macro | 52.9878% |
| ASTD | Sentiment Analysis | MARBERT | F1 macro | 58.0925% |
| ASTD | Sentiment Analysis | AraBERT | F1 macro | 57.0359% |
| HARD | Sentiment Analysis | AraBERT | F1 macro | 80.6464% |
| HARD | Sentiment Analysis | MARBERT | F1 macro | 80.3143% |
| ArSarcasm | Social Meaning | MARBERT | F1 macro | 76.6152% |
| ArSarcasm | Social Meaning | AraBERT | F1 macro | 74.9172% |

## Track B Dataset Results

| Dataset | Category | Model | Metric | Score | Correct | Total |
|---|---|---|---|---:|---:|---:|
| Arabic MMLU | Commonsense / STEM | Falcon | Accuracy | 29.6230% | 4282 | 14455 |
| Arabic MMLU | Commonsense / STEM | Jais | Accuracy | 26.6067% | 3846 | 14455 |
| Arabic MMLU | Commonsense / STEM | ALLAM | Accuracy | 0.0000% | 0 | 14455 |
| Arabic MMLU | Commonsense / STEM | Qwen | Accuracy | 0.0000% | 0 | 14455 |
| EXAMS AR | Commonsense / STEM | ALLAM | Accuracy | 53.4451% | 287 | 537 |
| EXAMS AR | Commonsense / STEM | Qwen | Accuracy | 48.9758% | 263 | 537 |
| EXAMS AR | Commonsense / STEM | Falcon | Accuracy | 21.0428% | 113 | 537 |
| EXAMS AR | Commonsense / STEM | Jais | Accuracy | 0.1862% | 1 | 537 |
| Madinah QA | Commonsense / STEM | Falcon | Accuracy | 26.1003% | 255 | 977 |
| Madinah QA | Commonsense / STEM | ALLAM | Accuracy | 0.0000% | 0 | 977 |
| Madinah QA | Commonsense / STEM | Qwen | Accuracy | 0.0000% | 0 | 977 |
| ArabCulture | Cultural Understanding | Falcon | Accuracy | 33.7284% | 370 | 1097 |
| ArabCulture | Cultural Understanding | Jais | Accuracy | 33.4549% | 367 | 1097 |
| ArabCulture | Cultural Understanding | ALLAM | Accuracy | 0.0000% | 0 | 1097 |
| ArabCulture | Cultural Understanding | Qwen | Accuracy | 0.0000% | 0 | 1097 |
| AraDiCE | Cultural Understanding | ALLAM | Accuracy | 0.0000% | 0 | 120 |
| AraDiCE | Cultural Understanding | Falcon | Accuracy | 0.0000% | 0 | 120 |
| AraDiCE | Cultural Understanding | Jais | Accuracy | 0.0000% | 0 | 120 |
| AraDiCE | Cultural Understanding | Qwen | Accuracy | 0.0000% | 0 | 120 |
| AraTrust | Social Meaning | Jais | Accuracy | 32.1839% | 168 | 522 |
| AraTrust | Social Meaning | Falcon | Accuracy | 26.6284% | 139 | 522 |
| AraTrust | Social Meaning | ALLAM | Accuracy | 25.0958% | 131 | 522 |
| AraTrust | Social Meaning | Qwen | Accuracy | 23.7548% | 124 | 522 |

## Model-Level Findings

### MARBERT

- Overall rank: 1
- Overall average: 68.2126%
- Categories completed: 4
- Best category: Social Meaning at 76.6152%
- Track A only in the compiled results.

### AraBERT

- Overall rank: 2
- Overall average: 66.7766%
- Categories completed: 4
- Best category: Social Meaning at 74.9172%
- Track A only in the compiled results.

### Falcon

- Overall rank: 3
- Overall average: 23.0271%
- Categories completed: 3
- Track B datasets completed: Arabic MMLU, Madinah QA, EXAMS AR, ArabCulture, AraDiCE, AraTrust
- Falcon-specific issue fixed: `'DynamicCache' object is not subscriptable`
- Final Falcon fix: native Transformers Falcon loading with compatible cache handling
- Falcon failed samples after rerun: 0
- Falcon-only full findings: `outputs/falcon/falcon_all_findings.md`

### Jais

- Overall rank: 4
- Overall average: 20.7693%
- Categories completed: 3
- Best category: Social Meaning at 32.1839%
- Missing expected result file: Track B Madinah QA
- Jais results were not rerun while preparing this consolidated file.

### ALLAM

- Overall rank: 5
- Overall average: 14.3036%
- Categories completed: 3
- Best category: Social Meaning at 25.0958%
- Track B only in the compiled results.

### Qwen

- Overall rank: 6
- Overall average: 13.3600%
- Categories completed: 3
- Best category: Social Meaning at 23.7548%
- Track B only in the compiled results.

## Runs Needing Attention

- No compiled run is currently marked invalid or missing-score.
- One expected result file is missing:

| Track | Dataset | Model |
|---|---|---|
| Track B | Madinah QA | Jais |

## Falcon Cache-Fix Summary

- Previous Falcon failure: `'DynamicCache' object is not subscriptable`
- Raw traceback: `logs/falcon_dynamic_cache_error_before_fix.log`
- Final fix used: loaded Falcon through the native Transformers implementation, avoiding the legacy remote Falcon code path that indexed the new cache object as a tuple.
- Falcon generation cache setting after fix: `use_cache=True` with native Transformers cache handling
- Transformers version: `5.4.0`
- Device used for Falcon rerun: `NVIDIA GeForce RTX 4060`
- Falcon raw outputs: `outputs/falcon/falcon_raw_outputs.jsonl`
- Falcon scores: `outputs/falcon/falcon_scores.csv` and `outputs/falcon/falcon_scores.json`

## Validation Notes

- Existing compiled results contain 33 valid scored runs.
- Invalid or missing-score compiled runs: 0
- Falcon raw outputs are non-empty and contain 17,708 evaluated examples.
- Falcon CSV and JSON scores agree.
- Falcon failed sample count is 0.
- This consolidated findings file was created from existing result artifacts only.
- No evaluations were rerun to create this file.
