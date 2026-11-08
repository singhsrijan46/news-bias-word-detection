# Bias Lens: whose bias? Bias-word detection on the Arduino UNO Q

> A project for the course "Trustworthy Artificial Intelligence", built on **Spinde et al. (2021)**, "Automated identification of bias inducing words in news articles using linguistic and context-oriented features" (*Information Processing and Management*), and its MBIC dataset.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ipm.2021.102505-blue)](https://doi.org/10.1016/j.ipm.2021.102505)

The paper's detector turns about ten crowd judgements per sentence into one yes/no label per word. Bias Lens keeps the votes instead:
- **How many readers:** it says how many readers in ten would flag each word, and marks the words that readers split on.
- **Whose readers:** it checks whose readers those are, and how the models treat each political side.
- **Where it runs:** on a US$79 Arduino UNO Q at home, so no article text leaves the house.

## What it does

1. **Send an article.** A phone or laptop on the home network sends an article to the board. It can use the board's web page, the Chrome or Edge extension, or a bookmark in Chrome on a phone. Each device pairs once with a code shown on the board's LED matrix. After that, every request is encrypted.
2. **The board scores every word.** A small distilled model runs on its CPU and marks each word with "about N in 10 readers would flag this". Flags are shown in two bands, "readers split" and "most readers would flag it". The LED matrix shows a heat strip of the article.
3. **On request, the two political sides.** A second model estimates how liberal and how conservative readers would react to each word, with the audit's caveat.
4. **On request, why.** For a selected word, a small language model on the board writes one sentence on why readers might find it one-sided. It is labelled experimental, with its measured record.

See `docs/BIAS_LENS_APP.md` and `docs/BIAS_LENS_CLIENTS.md`.

## Results at a glance

| Question | Finding | Details |
| --- | --- | --- |
| **RQ1, contested words.** Can a model learn how many readers would flag a word, without losing detection? | Yes. Trained on vote shares, RoBERTa matches the yes/no model (F1 0.617 against 0.608) and its shares are six times better calibrated. Its uncertainty marks contested words: setting aside its least certain 10% of words raises F1 to 0.70, against 0.66 for the yes/no model. | `docs/SOFT_LABEL_MODELS.md` |
| **RQ2, whose bias.** Whose judgement do the models reproduce, and do they treat the sides alike? | They follow liberal annotators more closely (Spearman's rho 0.48 against 0.40, with groups of equal size), because the annotator pool leans liberal. Party and politician names are treated almost alike, but "liberal" is flagged more than "conservative". The authors' own sentence classifiers rate a sentence as more biased when it names Trump instead of Biden; ours do not. | `docs/AUDIT.md` |
| **RQ3, the cost of the edge.** What do distillation and INT8 cost? | The board's MiniLM model in INT8 reaches F1 0.601, against 0.617 for its teacher. It handles an 850-word article in 1.3 seconds on the CPU. INT8 is faster than FP32 on this CPU, and the GPU is slower still. Compression costs a little political symmetry. The board gives the same scores as the PC. | `docs/EDGE_BENCHMARK.md`, `docs/STUDENT_MODEL.md` |
| **Transfer.** Does it hold beyond MBIC? | On BABE, which experts labelled, the models rank words as well as on MBIC (ROC AUC 0.95), including on topics MBIC lacks. Holding out each MBIC topic in turn costs less than 0.01 in F1. Experts mark fewer words, so at MBIC's threshold the models over-flag. | `docs/BABE_TEST.md` |
| **Privacy.** Does text stay home? | In a network log taken while the board analysed articles, the app talked to nothing but its own clients. | `docs/NETWORK_LOG.md` |
| **RQ4, the display.** Do readers judge bias better with vote shares? | Not run. It needs a reader study with ethics approval. | |

For comparison, the paper's XGBoost reaches an F1 of 0.43. Our sentence-grouped rerun of it reaches 0.45.

## Documents

Read in this order:

| Document | What it covers |
| --- | --- |
| `docs/PERSPECTIVE_ANALYSIS.md` | What MBIC's raw votes show: how contested words are, and how annotators' politics changes what they flag |
| `docs/SOFT_LABEL_MODELS.md` | M0 to M2: the paper's XGBoost, and RoBERTa on yes/no labels and on vote shares. Also abstaining, and topics held out (RQ1) |
| `docs/STUDENT_MODEL.md` | Distilling the vote-share model into the board's MiniLM student, and what INT8 costs |
| `docs/EDGE_BENCHMARK.md` | Speed and memory on the UNO Q: seven encoders, CPU against GPU, and the board against the PC (RQ3) |
| `docs/AUDIT.md` | Political symmetry, outlet leaning, whose readers, compression, and the authors' classifiers (RQ2, RQ3) |
| `docs/BABE_TEST.md` | The external test against expert labels |
| `docs/BIAS_LENS_APP.md` | The board's app: page, API, LED matrix and deployment |
| `docs/BIAS_LENS_CLIENTS.md` | The extension, the phone bookmark, and pairing and encryption |
| `docs/EXPLANATIONS.md` | The on-board language model that explains one word: how it is kept honest, and how good it is |
| `docs/NETWORK_LOG.md` | The evidence that no text leaves the board |
| `docs/MODEL_CARD.md` | The board's model: intended use, results and caveats |
| `docs/DATASHEET.md` | The released label set: MBIC's votes by annotator ideology |
| `docs/PAPER_REPRODUCTION.md` | Phase 1: the team's reproduction of the paper (`model_comparison/`, `optimized_model/`) |

## Repository map

```
app/bias-lens/        the App Lab app: web page, Python API and analyzer, LED matrix sketch
clients/              the Chrome and Edge extension, and the pairing, encryption and overlay code it shares with the board
scripts/              training, evaluation and audits; edge/: the board benchmark; app/: deployment and the network log
src/                  data loading, word targets, models and metrics
data/raw/             MBIC as released, without the MTurk IDs
data/derived/         the released label set (docs/DATASHEET.md)
docs/                 one document per step, listed above
tests/                the pytest suite
model_comparison/, optimized_model/   phase 1: the paper reproduction
```

## Quick start

```bash
pip install -r requirements.txt -r requirements-transformers.txt
python -m pytest
```

Train and evaluate. A GPU helps: our runs used a 4 GB laptop GPU.

```bash
python scripts/train_token_models.py          # M0 to M2, 10 sentence-grouped folds (about 30 minutes)
python scripts/train_student.py               # the distilled student, cross-validated (about an hour)
python scripts/train_student.py --final       # the final student and teacher, for the board and the audits
python scripts/train_perspective_model.py --final   # M3, the liberal and conservative model
python scripts/audit_models.py                # the audit (docs/AUDIT.md)
python scripts/evaluate_babe.py               # the external test (docs/BABE_TEST.md)
```

Put it on an Arduino UNO Q, connected over USB-C and set up in Arduino App Lab:

```bash
python scripts/app/fetch_llm.py          # the explainer's language model and llama.cpp, 1.1 GB (or install --no-llm)
python scripts/app/deploy_app.py install
python scripts/app/deploy_app.py start
```

Then open `http://<board name>.local:7000` on any device on the same network, and pair it with the code on the LED matrix.

## Citation

The project builds on this paper and its dataset:

```bibtex
@article{spinde2021automated,
  title={Automated identification of bias inducing words in news articles using linguistic and context-oriented features},
  author={Spinde, Timo and Rudnitckaia, Lada and Mitrovi{\'c}, Jelena and Hamborg, Felix and Granitzer, Michael and Gipp, Bela and Donnay, Karsten},
  journal={Information Processing \& Management},
  volume={58},
  number={3},
  pages={102505},
  year={2021},
  publisher={Elsevier}
}
```
