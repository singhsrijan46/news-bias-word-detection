# Bias Lens: the reading aid on the Arduino UNO Q

Bias Lens is the project's device. Any phone or laptop on the home network can send it news text. It marks the words that readers would call biased and says how many readers in ten would flag each one. Everything runs on the board: no text goes to a server, and nothing is stored.

## How it fits together

| Part | Where | What it does |
| --- | --- | --- |
| Web page (`app/bias-lens/assets/`) | Any browser on the network | Paste text or send it with a bookmark, then see the marked-up article and a summary |
| API (`python/main.py`) | Board, Linux side, inside App Lab's Python container | `POST /api/analyze` and `GET /api/info` on port 7000, served by App Lab's WebUI brick |
| Analyzer (`python/analyzer.py`) | Board, Linux side | Splits the text into sentences, scores every word, assigns bands, builds the summary and the heat strip |
| Model (`python/onnx_token_model.py`, `python/model/`) | Board, Linux side, on the CPU | The distilled MiniLM-L6 student in INT8, run with ONNX Runtime |
| Perspective model (`python/perspective/`) | Board, Linux side, on the CPU, on request | M3 in INT8: how MBIC's liberal and conservative annotators would each react to a word. It is loaded on first use. |
| Sketch (`sketch/sketch.ino`) | Board, STM32 microcontroller | Draws the heat strip, a scanning line while busy, and an idle icon on the 8 × 13 LED matrix |

The Linux side tells the sketch what to show over App Lab's Bridge, using `show_busy`, `show_levels` and `show_idle`.

The model choices follow the benchmark (`docs/EDGE_BENCHMARK.md`):
- **INT8:** faster and smaller than FP32. In cross-validation the INT8 student's F1 was 0.601 against 0.605, which is not a significant difference (`docs/STUDENT_MODEL.md`).
- **One sentence at a time:** padded batches were slower on this CPU.
- **CPU, not GPU:** on the GPU, FP16 gave wrong results and FP32 was 10 to 24 times slower.

## What the page shows

A word's **share** is the model's estimate of the fraction of readers who would flag it. The model was distilled from a RoBERTa teacher trained on MBIC's vote shares (`docs/SOFT_LABEL_MODELS.md`).

| Band | Share | Shown as |
| --- | --- | --- |
| Not flagged | Below the flag threshold | Plain text |
| Readers split (contested) | From the threshold to just under 0.6 | Amber, with a dashed underline |
| Most readers would flag it | 0.6 or more | Red |

**The flag threshold.** It is the student's best-F1 threshold from cross-validation, averaged over the folds and stored in `model_info.json`. Tooltips round shares to "about N in 10 readers".

**The heat strip.** It cuts the article into 13 stretches of sentences. Each column is as tall as its stretch's most-flagged word, in eighths, and is empty when nothing in that stretch is flagged. The LED matrix shows the same strip.

**The lexicon.** Words also found in the paper's bias lexicon (`bised_words_lexicon.xlsx` and `seed_biased_words.xlsx`, 2,216 words) are noted in the tooltip, as a second, independent piece of evidence.

**Liberal and conservative readers, on request.** After an analysis, **Compare them** asks the board's perspective model, M3 (`docs/AUDIT.md`, section 3). M3 estimates the share of MBIC's liberal annotators (ideology below −3) and of its conservative ones (above 3) who would flag each word.
- **Tooltips:** each marked word's tooltip gains both estimates.
- **Lists:** the page lists the words where the two groups differ by at least 1 reader in 10, among words that either group might flag.
- **Caveat:** M3 fits conservative annotators better than the main model's pooled share. But its gap between the groups follows the observed gap only weakly (Spearman's rho 0.13), so the page says that a single word's difference is uncertain.
- **Speed:** M3 is RoBERTa-base in INT8. On the board it took 6.9 seconds for an 820-word article, or 10 seconds on the first request, which also loads it. The main analysis of the same text took 1.3 seconds.

**Whose readers.** The page's "How to read this" says that MBIC's annotators lean liberal (44% liberal, 27% conservative). It also says that a neutral article still shows some flags, mostly "readers split": in the BABE test, experts marked about a third of those words (`docs/BABE_TEST.md`).

## API

```http
POST /api/analyze
Content-Type: application/json

{"text": "Critics slammed the reckless plan."}
```

The response gives:
- `text`: the text that was analysed. Input is cut at 20,000 characters, about 3,000 words.
- `sentences`: each sentence's character span and its words. Each word comes with its span, `share`, `band` and `lexicon`; words with a share below 0.05 are left out.
- `summary`: counts, the 13-column `strip`, the most-flagged words, and the time taken.

`GET /api/info` describes the model, the threshold and the cross-validated scores, and says whether the perspective model is installed (`perspective`).

Paired devices send these requests encrypted to `POST /api/secure` (`docs/BIAS_LENS_CLIENTS.md`):
- **`op: "analyze"`:** the analysis above.
- **`op: "perspective"`:** M3's view of the same text, for each word either group might flag:
  - `words`: each word's span, `liberal` and `conservative` shares.
  - `liberal_more` and `conservative_more`: up to six words each, where one group's share is at least 0.1 higher.
  - `strip`: the heat strip from M3's pooled output, for the LED matrix.

**Other devices.** The page's "Send text from your other devices" panel has a bookmarklet. It opens Bias Lens with the selected text, or the whole page, in the address's `#` part, which browsers never send over the network, and the page then posts it to the board.

## Deploy

On the PC, with the board connected over USB-C:

```bash
python scripts/train_student.py --final           # trains and exports the student (outputs/student/final)
python scripts/train_perspective_model.py --final # trains and exports M3 (outputs/models/perspective_final)
python scripts/app/deploy_app.py install          # builds the app folder and copies it to /home/arduino/ArduinoApps
python scripts/app/deploy_app.py start            # starts it with App Lab's CLI; also compiles and flashes the sketch
python scripts/app/deploy_app.py forward          # optional: http://localhost:7000 on this PC
```

On the home network, the page is at `http://<board name>.local:7000`. It also appears in App Lab as "Bias Lens".

Notes from setting this up on our board:
- **Libraries.** App Lab installs an app's `requirements.txt` through Docker's default network, where large downloads stalled on our Wi-Fi. So `deploy_app.py` installs `onnxruntime` and `tokenizers` once, over the board's own network, into `/home/arduino/.cache/bias-lens/vendor`, and copies them into the app. numpy comes with App Lab's image.
- **Temp folder.** `adb` sessions on the board set `TMPDIR` to Android's `/data/local/tmp`, which does not exist there, and App Lab's CLI then fails with "stat /data/local/tmp". `deploy_app.py` runs the CLI with `TMPDIR=/tmp`.
- **The perspective model.** `install` copies M3 into the app when `outputs/models/perspective_final` exists. It adds 125 MB to the app. With both models loaded, the app's container used 319 MB of the board's 3.6 GB. `--no-perspective` leaves it out, and the page then hides the button.

## Trustworthy AI, by design

- **Privacy:** text is analysed on the board and kept only for the request. There are no accounts, no cloud and no logs of text; the app prints only word counts.
- **Honest uncertainty:** the page shows how many readers would agree, not a bare "biased", and marks contested words as contested.
- **Transparency:** the page explains where the scores come from and their limits, and shows the model and its cross-validated quality.
- **Whose readers:** the page says the annotators lean liberal, and on request shows liberal and conservative readers apart, with the caveat the audit calls for.
- **Human agency:** a flag is a prompt to look again. The reader decides.

## Limits

- **Narrow training data.** The model learned from MBIC: English, US political news from 2017 to 2020, and US crowd workers. It may miss new terms, and it reflects those annotators.
- **Words, not stance.** It scores words, not facts or framing beyond the sentence.
- **The ideology view is uncertain.** M3 tells the two groups apart only weakly, as the page says. Whether such a view helps readers is RQ4's question, which needs a reader study.
- **Only on this page.** The extension and the phone bookmark show the main view; the liberal and conservative estimates are on the board's own page.
