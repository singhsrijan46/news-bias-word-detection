# Bias Lens: the reading aid on the Arduino UNO Q

Bias Lens is the project's device. Any phone or laptop on the home network can send it news text. It marks the words that readers would call biased and says how many readers in ten would flag each one. Everything runs on the board: no text goes to a server, and nothing is stored.

## How it fits together

| Part | Where | What it does |
| --- | --- | --- |
| Web page (`app/bias-lens/assets/`) | Any browser on the network | Paste text or send it with a bookmark, then see the marked-up article and a summary |
| API (`python/main.py`) | Board, Linux side, inside App Lab's Python container | `POST /api/analyze` and `GET /api/info` on port 7000, served by App Lab's WebUI brick |
| Analyzer (`python/analyzer.py`) | Board, Linux side | Splits the text into sentences, scores every word, assigns bands, builds the summary and the heat strip |
| Model (`python/onnx_token_model.py`, `python/model/`) | Board, Linux side, on the CPU | The distilled MiniLM-L6 student in INT8, run with ONNX Runtime |
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

`GET /api/info` describes the model, the threshold and the cross-validated scores.

**Other devices.** The page's "Send text from your other devices" panel has a bookmarklet. It opens Bias Lens with the selected text, or the whole page, in the address's `#` part, which browsers never send over the network, and the page then posts it to the board.

## Deploy

On the PC, with the board connected over USB-C:

```bash
python scripts/train_student.py --final           # trains and exports the student (outputs/student/final)
python scripts/app/deploy_app.py install          # builds the app folder and copies it to /home/arduino/ArduinoApps
python scripts/app/deploy_app.py start            # starts it with App Lab's CLI; also compiles and flashes the sketch
python scripts/app/deploy_app.py forward          # optional: http://localhost:7000 on this PC
```

On the home network, the page is at `http://<board name>.local:7000`. It also appears in App Lab as "Bias Lens".

Notes from setting this up on our board:
- **Libraries.** App Lab installs an app's `requirements.txt` through Docker's default network, where large downloads stalled on our Wi-Fi. So `deploy_app.py` installs `onnxruntime` and `tokenizers` once, over the board's own network, into `/home/arduino/.cache/bias-lens/vendor`, and copies them into the app. numpy comes with App Lab's image.
- **Temp folder.** `adb` sessions on the board set `TMPDIR` to Android's `/data/local/tmp`, which does not exist there, and App Lab's CLI then fails with "stat /data/local/tmp". `deploy_app.py` runs the CLI with `TMPDIR=/tmp`.

## Trustworthy AI, by design

- **Privacy:** text is analysed on the board and kept only for the request. There are no accounts, no cloud and no logs of text; the app prints only word counts.
- **Honest uncertainty:** the page shows how many readers would agree, not a bare "biased", and marks contested words as contested.
- **Transparency:** the page explains where the scores come from and their limits, and shows the model and its cross-validated quality.
- **Human agency:** a flag is a prompt to look again. The reader decides.

## Limits

- **Narrow training data.** The model learned from MBIC: English, US political news from 2017 to 2020, and US crowd workers. It may miss new terms, and it reflects those annotators.
- **Words, not stance.** It scores words, not facts or framing beyond the sentence.
- **No ideology breakdown yet.** The liberal-versus-conservative breakdown from the proposal (RQ4) needs the perspective model, M3, which is not trained yet.
