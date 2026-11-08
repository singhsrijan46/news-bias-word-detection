# Model card: the Bias Lens word model

This card follows Mitchell et al. (2019), "Model Cards for Model Reporting". It covers the model that Bias Lens runs on the Arduino UNO Q, and briefly the two models around it. The numbers come from the project's own tests, linked in each section.

## Model details

| | The board's model | Its teacher | The perspective model (M3) |
| --- | --- | --- | --- |
| What it is | MiniLM-L6 token classifier (`nreimers/MiniLM-L6-H384-uncased`), distilled from the teacher | RoBERTa-base token classifier: M2 | RoBERTa-base with three outputs per word |
| Parameters | 22.6M | 124M | 124M |
| Output per word | Estimated share of readers who would flag the word as biased | The same | Shares of all, of liberal and of conservative annotators |
| Runs | On the board, INT8, ONNX Runtime, 4 CPU threads | Training and audits only | On the board, INT8, on request |
| File | 23 MB in INT8 (`outputs/student/final/model_int8.onnx`) | 497 MB in FP32 | 125 MB in INT8 |

- **Training:**
  - **Teacher:** trained on MBIC's vote shares (`docs/SOFT_LABEL_MODELS.md`).
  - **Student:** learned from those shares and from the teacher's scores on 29,479 unannotated sentences from the same articles (`docs/STUDENT_MODEL.md`).
  - **INT8:** ONNX Runtime's dynamic quantization.
- **Decision rule:**
  - **Flags:** a word is flagged when its share reaches 0.267, the INT8 student's best-F1 threshold averaged over 10 folds.
  - **Bands:** a flagged share below 0.6 is shown as "readers split", and 0.6 or more as "most readers would flag it".
- **Version:** trained on 2026-11-08 from this repository; `model_info.json` stores the settings and the cross-validated scores.
- **Not part of the scoring:** on request, a small general-purpose language model on the board, Qwen2.5-0.5B-Instruct, writes one sentence on why a word might read as one-sided. It does not change any score. Its reasons were sensible about a third of the time (`docs/EXPLANATIONS.md`).
- **Licence and data terms:** the repository's README gives the code an MIT licence. The model is trained on MBIC, whose authors' terms apply to the data.

## Intended use

- **Primary use:** a reading aid. While a person reads news, the device marks words that readers would call biased, and says how many readers in ten would. The aim is to prompt a second look and build media literacy.
- **Primary users:** readers on their own devices, on the home network of a board they own.
- **Out of scope:**
  - **Judging facts:** whether a sentence is true.
  - **Verdicts on outlets or authors:** rating them, ranking them, or moderating, filtering or blocking their content.
  - **Automated decisions about people.**
  - **Other languages and countries,** and text far from news.

## Factors

The model was checked for the factors where a bias detector is most likely to be unfair:
- **Political side of the text:** counterfactual swaps of party, politician and label words.
- **Outlet leaning:** left, centre and right outlets, as rated by AllSides.
- **Annotators' ideology:** liberal and conservative annotators' own shares.
- **Topic:** each topic held out in turn, and BABE's topics that MBIC lacks.
- **Compression:** FP32 against INT8, and the PC against the board.

## Metrics

- **Detection:** F1, precision and recall at the paper's label (at least 4 of about 10 votes), and ROC AUC. Accuracy is never reported: with 9.5% biased words, flagging nothing scores 90.5%.
- **Calibration:** share ECE, the gap between predicted and actual vote shares.
- **Symmetry:** score differences between the two sides of each swapped pair, and flags that change on the other words.
- **Expert agreement:** F1 and Cohen's κ against BABE's expert labels.

## Training and evaluation data

- **Training:** MBIC's 1,700 sentences, about 10 crowd judgements each, from 8 US outlets and 14 topics, 2017 to 2020. Also 29,479 unannotated sentences from the same 918 articles, scored by the teacher. See `docs/DATASHEET.md`.
- **Evaluation:**
  - **MBIC:** 10-fold cross-validation grouped by sentence, and 14 folds with each topic held out (M1 and M2).
  - **BABE:** 2,418 sentences not in MBIC, with expert labels (`docs/BABE_TEST.md`).
  - **Counterfactuals:** 4,631 sentences rewritten to name the other political side (`docs/AUDIT.md`).

## Quantitative analyses

**Detection**

| Test | The board's model (INT8) | Notes |
| --- | --- | --- |
| MBIC, 10-fold cross-validation | F1 0.601, ROC AUC 0.932, share ECE 0.015 | FP32 student: F1 0.605; teacher: 0.617; the paper's XGBoost: 0.43 |
| MBIC, holding out the 10% least certain words | F1 0.684 on the rest | `docs/SOFT_LABEL_MODELS.md` |
| BABE's new sentences, against experts | F1 0.516, ROC AUC 0.945 | At a threshold chosen on BABE: F1 0.55 |
| On the board, against the PC | 44 of 127,771 flags differ | Same F1 against BABE's experts, to four decimals (`docs/EDGE_BENCHMARK.md`) |

**Fairness and symmetry** (`docs/AUDIT.md`, `docs/BABE_TEST.md`)

| Check | Result |
| --- | --- |
| Swapped party and politician names | Scored almost alike: Biden 0.018 against Trump 0.024, Democrats 0.056 against Republicans 0.042 |
| Swapped labels | "Liberal" 0.279 against "conservative" 0.140, the largest asymmetry |
| Other words of a swapped sentence | 0.57% change flag, against 0.31% for the teacher: compression costs a little symmetry |
| Outlet leaning, MBIC | F1 0.622 left, 0.526 centre, 0.596 right; centre outlets have fewer biased words |
| Centre outlets, BABE's experts | Precision 0.11: most flags there are words experts did not mark, and 91% of them are "readers split" |
| Whose readers | Predicted shares follow liberal annotators more closely than conservative ones (Spearman's rho 0.48 against 0.40 for groups of equal size) |

**Speed on the UNO Q:** 41 ms per sentence and about 1.3 seconds for an 850-word article, using about 185 MB of memory (`docs/EDGE_BENCHMARK.md`).

## Ethical considerations

- **Whose judgement.** The model reproduces MBIC's crowd, and that crowd leans liberal: 44% liberal and 27% conservative. A flag means "many of these readers would flag this", not "this is biased". The page says so, and offers M3's liberal and conservative estimates on request.
- **The word "liberal".** In identical sentences it is flagged about twice as often as "conservative". The annotators treated it as a loaded label, and the model learned the word.
- **Over-trust.** A highlighted word can read as an accusation. Shares, the "readers split" band and the page's explanation are there to make flags read as prompts, not verdicts.
- **Neutral outlets.** Against expert labels, most flags on centre outlets are false. They are mostly "readers split", but they could still make a neutral article look biased.
- **Privacy.** All analysis runs on the board, and no text leaves it (`docs/NETWORK_LOG.md`). No annotator can be identified from the model or the released labels.

## Caveats and recommendations

- **Use it as a prompt, not a verdict.** Show shares and bands, never a bare "biased".
- **Stay close to the training data.** It covers US political news in English from 2017 to 2020. New terms and other countries' politics are not covered.
- **Re-run the swap test before each release.** Compression and retraining can move individual pairs, as INT8 did for left-wing and right-wing.
- **Confirm small differences with more seeds.** One seed was trained for every model.
- **The perspective model separates the groups only weakly.** Its gap between groups follows the observed gap at Spearman's rho 0.13. Show its two estimates with that caveat, as the page does.

## References

- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., and Gebru, T. (2019). Model cards for model reporting. In *Proceedings of the Conference on Fairness, Accountability, and Transparency* (FAT* '19), 220–229.
- Spinde, T., Rudnitckaia, L., Mitrović, J., Hamborg, F., Granitzer, M., Gipp, B., and Donnay, K. (2021). Automated identification of bias inducing words in news articles using linguistic and context-oriented features. *Information Processing and Management*, 58, 102505.
