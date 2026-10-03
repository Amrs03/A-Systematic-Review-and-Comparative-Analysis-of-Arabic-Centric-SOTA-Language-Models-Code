> [!WARNING]
> **Unaudited provenance snapshot — preserved verbatim, not a result of this study.**
>
> This file is the compiled output of the original run, kept unchanged so the audit can be
> checked against it. The leaderboard below is precisely the ranking that
> [`AUDIT_REPORT.md`](../AUDIT_REPORT.md) invalidates: 21 of the 23 Track B cells that feed
> these averages are harness artifacts, not capability measurements. Do not cite these
> numbers. Image links point at files that were not preserved.

# Arabic NLP Evaluation Results

Generated: 2026-05-30 05:05 (Falcon rows updated only)

This report compiles all JSON files found under `results/track_a`, `results/track_b`, and `results/bridge`.

Important reading note: scores are averaged only across categories where a model has a valid completed result. Track A and Track B evaluate different model families and tasks, so use the per-track and per-category tables for the fairest comparisons.

## Snapshot

| Item                            | Value                                       |
| ------------------------------- | ------------------------------------------- |
| Result files read               | 33                                          |
| Valid scored runs               | 33                                          |
| Invalid / missing-score runs    | 0                                           |
| Models found in files           | ALLAM, AraBERT, Falcon, Jais, MARBERT, Qwen |
| Models with valid ranked scores | MARBERT, AraBERT, Falcon, Jais, ALLAM, Qwen |

## Overall Leaderboard

| Rank | Model   | Average | Categories |
| ---- | ------- | ------- | ---------- |
| 1    | MARBERT | 68.2%   | 4          |
| 2    | AraBERT | 66.8%   | 4          |
| 3    | Falcon  | 23.0%   | 3          |
| 4    | Jais    | 20.8%   | 3          |
| 5    | ALLAM   | 14.3%   | 3          |
| 6    | Qwen    | 13.4%   | 3          |

## Visuals

![Overall Leaderboard](visuals/overall_leaderboard.png)

![Category Heatmap](visuals/category_heatmap.png)

![Track A Dataset Scores](visuals/track_a_dataset_scores.png)

![Track B Dataset Scores](visuals/track_b_dataset_scores.png)

![Evaluation Errors](visuals/evaluation_errors.png)

## Category Scores

| Category                 | Model   | Average | Datasets                          |
| ------------------------ | ------- | ------- | --------------------------------- |
| Commonsense / STEM       | Falcon  | 25.6%   | Arabic MMLU, Madinah QA, EXAMS AR |
| Commonsense / STEM       | ALLAM   | 17.8%   | Arabic MMLU, EXAMS AR, Madinah QA |
| Commonsense / STEM       | Qwen    | 16.3%   | Arabic MMLU, EXAMS AR, Madinah QA |
| Commonsense / STEM       | Jais    | 13.4%   | Arabic MMLU, EXAMS AR             |
| Cultural Understanding   | Falcon  | 16.9%   | ArabCulture, AraDiCE              |
| Cultural Understanding   | Jais    | 16.7%   | AraDiCE, ArabCulture              |
| Cultural Understanding   | ALLAM   | 0.0%    | AraDiCE, ArabCulture              |
| Cultural Understanding   | Qwen    | 0.0%    | AraDiCE, ArabCulture              |
| Dialect Identification   | MARBERT | 74.0%   | QADI                              |
| Dialect Identification   | AraBERT | 68.6%   | QADI                              |
| Named Entity Recognition | AraBERT | 54.7%   | ANERcorp                          |
| Named Entity Recognition | MARBERT | 53.0%   | ANERcorp                          |
| Sentiment Analysis       | MARBERT | 69.2%   | ASTD, HARD                        |
| Sentiment Analysis       | AraBERT | 68.8%   | ASTD, HARD                        |
| Social Meaning           | MARBERT | 76.6%   | ArSarcasm                         |
| Social Meaning           | AraBERT | 74.9%   | ArSarcasm                         |
| Social Meaning           | Jais    | 32.2%   | AraTrust                          |
| Social Meaning           | Falcon  | 26.6%   | AraTrust                          |
| Social Meaning           | ALLAM   | 25.1%   | AraTrust                          |
| Social Meaning           | Qwen    | 23.8%   | AraTrust                          |

## Dataset-Level Results

| Track   | Category                 | Dataset     | Model   | Metric   | Score | Total |
| ------- | ------------------------ | ----------- | ------- | -------- | ----- | ----- |
| Track A | Dialect Identification   | QADI        | MARBERT | F1 macro | 74.0% |       |
| Track A | Dialect Identification   | QADI        | AraBERT | F1 macro | 68.6% |       |
| Track A | Named Entity Recognition | ANERcorp    | AraBERT | F1 macro | 54.7% |       |
| Track A | Named Entity Recognition | ANERcorp    | MARBERT | F1 macro | 53.0% |       |
| Track A | Sentiment Analysis       | ASTD        | MARBERT | F1 macro | 58.1% |       |
| Track A | Sentiment Analysis       | ASTD        | AraBERT | F1 macro | 57.0% |       |
| Track A | Sentiment Analysis       | HARD        | AraBERT | F1 macro | 80.6% |       |
| Track A | Sentiment Analysis       | HARD        | MARBERT | F1 macro | 80.3% |       |
| Track A | Social Meaning           | ArSarcasm   | MARBERT | F1 macro | 76.6% |       |
| Track A | Social Meaning           | ArSarcasm   | AraBERT | F1 macro | 74.9% |       |
| Track B | Commonsense / STEM       | Arabic MMLU | Jais    | Accuracy | 26.6% | 14455 |
| Track B | Commonsense / STEM       | Arabic MMLU | ALLAM   | Accuracy | 0.0%  | 14455 |
| Track B | Commonsense / STEM       | Arabic MMLU | Qwen    | Accuracy | 0.0%  | 14455 |
| Track B | Commonsense / STEM       | EXAMS AR    | ALLAM   | Accuracy | 53.4% | 537   |
| Track B | Commonsense / STEM       | EXAMS AR    | Qwen    | Accuracy | 49.0% | 537   |
| Track B | Commonsense / STEM       | EXAMS AR    | Jais    | Accuracy | 0.2%  | 537   |
| Track B | Commonsense / STEM       | Madinah QA  | ALLAM   | Accuracy | 0.0%  | 977   |
| Track B | Commonsense / STEM       | Madinah QA  | Qwen    | Accuracy | 0.0%  | 977   |
| Track B | Cultural Understanding   | AraDiCE     | ALLAM   | Accuracy | 0.0%  | 120   |
| Track B | Cultural Understanding   | AraDiCE     | Jais    | Accuracy | 0.0%  | 120   |
| Track B | Cultural Understanding   | AraDiCE     | Qwen    | Accuracy | 0.0%  | 120   |
| Track B | Cultural Understanding   | ArabCulture | Jais    | Accuracy | 33.5% | 1097  |
| Track B | Cultural Understanding   | ArabCulture | ALLAM   | Accuracy | 0.0%  | 1097  |
| Track B | Cultural Understanding   | ArabCulture | Qwen    | Accuracy | 0.0%  | 1097  |
| Track B | Social Meaning           | AraTrust    | Jais    | Accuracy | 32.2% | 522   |
| Track B | Social Meaning           | AraTrust    | ALLAM   | Accuracy | 25.1% | 522   |
| Track B | Social Meaning           | AraTrust    | Qwen    | Accuracy | 23.8% | 522   |
| Track B | Commonsense / STEM | Arabic MMLU | Falcon | Accuracy | 29.6% | 14455 |
| Track B | Commonsense / STEM | Madinah QA | Falcon | Accuracy | 26.1% | 977 |
| Track B | Commonsense / STEM | EXAMS AR | Falcon | Accuracy | 21.0% | 537 |
| Track B | Social Meaning | AraTrust | Falcon | Accuracy | 26.6% | 522 |
| Track B | Cultural Understanding | ArabCulture | Falcon | Accuracy | 33.7% | 1097 |
| Track B | Cultural Understanding | AraDiCE | Falcon | Accuracy | 0.0% | 120 |

## Runs Needing Attention

| Track   | Dataset     | Model  | Status  | Errors | Note                                                                                            |
| ------- | ----------- | ------ | ------- | ------ | ----------------------------------------------------------------------------------------------- |

## Missing Result Files

| Track   | Dataset    | Model |
| ------- | ---------- | ----- |
| Track B | Madinah QA | Jais  |
