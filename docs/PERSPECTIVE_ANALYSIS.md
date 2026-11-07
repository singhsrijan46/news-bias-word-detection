# Perspective analysis of MBIC

This analysis reproduces the preliminary evidence in the "Whose Bias?" project proposal. It uses MBIC's raw per-annotator file (`data/raw/annotations.xlsx`), which records each annotator's highlights and political ideology.

## Run it

```bash
python scripts/run_perspective_analysis.py
```

- **Run time:** a few minutes, mostly 800 permutation runs.
- **Quick check:** `--quick` runs 20 shuffles per test. Use it only to check that everything runs; its p-values are too coarse to report.
- **Outputs:** saved to `outputs/perspective/`, which git ignores:
  - `perspective_summary.json`: every number below.
  - `word_rate_differences.csv`: per-word highlight rates for liberal and conservative annotators.

## What it does

| Step | Code | What it measures |
| --- | --- | --- |
| 1 | `src/data/annotations.py` | Rebuilds each annotator's highlights as token positions, and checks them against the released `biased_words4` labels |
| 2 | `contestedness_distribution` | The share of annotators behind each highlighted word |
| 3 | `scripts/run_perspective_analysis.py` | Biased words at vote cutoffs of 3, 4 and 5 (`label3` to `label5` in `dt_final`) |
| 4 | `ideology_word_choice_test`, `group_majority_tests` | Whether ideology predicts which words people highlight, by permutation tests |
| 5 | `word_rate_differences` | Per-word liberal vs conservative highlight rates, with Fisher exact tests and a Bonferroni correction |

## Conventions

- **Ideology scale:** `political_ideology` runs from -10 (liberal) to +10 (conservative).
- **Paper's grouping:** liberal below -3, centre -3 to 3, conservative above 3. This reproduces the paper's 44.3% / 29.1% / 26.7% split.
- **Simple split (step 5 only):** every negative score counts as liberal and every positive score as conservative. This gives more data per word, but the step is exploratory.
- **What the shuffles keep:** permutation tests shuffle ideology across annotators, so each annotator's own highlights stay intact.
- **Seeds:** fixed (3 for the continuous test, 2 for the group tests), so a full run reproduces the numbers below exactly.

## Results of a full run

| Check | Result |
| --- | --- |
| Rebuilt vs published 4-vote labels | Jaccard 0.967, precision 0.983, recall 0.968; 1,576 of 1,700 sentences match exactly |
| How contested highlights are | 16,955 of 56,909 tokens were highlighted by someone. 58.2% of those were picked by under 20% of annotators; 2.1% by 80% or more. |
| Biased words at 3 / 4 / 5 votes | 4,489 / 3,018 / 2,082 of 31,794 |
| Ideology predicts which words (continuous, all annotators) | z = 4.0 (p = 0.003); z = 3.1 (p = 0.007) with each annotator's overall rate removed; no overall direction (p = 0.16 and 0.27) |
| Paper's grouping, per-sentence majorities | Split words p = 0.53; opposite verdicts p = 0.22 (only 2 to 3 conservatives per sentence) |
| Centre outlets, conservative minus liberal | +0.40 highlighted words per sentence (p = 0.005) |
| Per-word rates | 537 words tested; *donald*, *trump* and *abortion* survive Bonferroni |
