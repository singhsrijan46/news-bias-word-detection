# On-device benchmark: can the UNO Q run a bias-word model?

The project's device workstream starts by timing candidate models on the Arduino UNO Q, before any student model is trained. This tells us how big the distilled student can be. It also tests the main hardware risk: the board's Cortex-A53 cores lack Arm's dot-product instructions, so 8-bit (INT8) models might run slower than full-precision (FP32) ones.

## What is measured

**Models.** Seven encoders, each with a per-token head giving one score per token: the shape the soft-label model and its student will have. The heads are untrained, because speed and memory do not depend on the weights.

| Name | Hugging Face model | Parameters | Layers × width |
| --- | --- | --- | --- |
| bert-tiny | google/bert_uncased_L-2_H-128_A-2 | 4.4M | 2 × 128 |
| bert-mini | google/bert_uncased_L-4_H-256_A-4 | 11.1M | 4 × 256 |
| electra-small | google/electra-small-discriminator | 13.5M | 12 × 256 |
| minilm-l6 | nreimers/MiniLM-L6-H384-uncased | 22.6M | 6 × 384 |
| bert-small | google/bert_uncased_L-4_H-512_A-8 | 28.5M | 4 × 512 |
| distilroberta | distilbert/distilroberta-base | 81.5M | 6 × 768 |
| roberta-base | FacebookAI/roberta-base | 124.1M | 12 × 768 |

**Workload.** Real MBIC text, the same for every model:
- 200 sentences, with a median of 42 tokens.
- 10 whole articles of 421 to 1,289 words, with a median of 854 words (43 sentences).

**CPU runs.** ONNX Runtime 1.28 inside App Lab's own Python image, so the environment is the one an App Lab app would run in.
- **Precision:** FP32, and INT8 from ONNX Runtime's dynamic quantization.
- **Threads:** 1 and 4.
- **Timing:** single sentences (batch size 1), and whole articles both sentence by sentence and as one padded batch.
- **Isolation:** each run is a fresh process, and the CPU cools to 50 °C before each run.

**GPU runs.** The board's Adreno 702 GPU is reachable through Mesa, in two ways: Vulkan 1.0 (Turnip) and OpenCL 3.0 (Rusticl). ONNX Runtime cannot use either, so the GPU runs use ncnn on Vulkan.
- **Precision:** ncnn computes in FP16 on GPUs that support it. Each model is also run with FP16 turned off.
- **Accuracy:** every output is compared with ncnn's full-FP32 result on the CPU.
- **Comparison:** ncnn's CPU mode on 4 threads gives a like-for-like baseline.
- **Inputs:** these models take token ids only, because one sentence at a time needs no padding mask.

**Also recorded:** load time, peak memory, CPU temperature and frequency, and on GPU runs the GPU's temperature and frequency, plus how many CPU cores the run kept busy.

## How to run

On the PC:

```bash
pip install -r scripts/edge/requirements.txt
python scripts/edge/export_encoders.py                 # ONNX, FP32 and INT8, about 7 minutes
python scripts/edge/export_ncnn.py                     # ncnn, for the GPU
```

With the board connected over USB-C and set up in App Lab:

```bash
python scripts/edge/run_on_board.py setup              # copy models, install ONNX Runtime on the board
python scripts/edge/run_on_board.py start              # the CPU benchmark, about an hour
python scripts/edge/run_on_board.py status
python scripts/edge/run_on_board.py gpu-setup          # copy ncnn models, install ncnn
python scripts/edge/run_on_board.py gpu-start          # the GPU benchmark
python scripts/edge/run_on_board.py fetch
python scripts/edge/summarize_benchmark.py
```

- **Shorter GPU runs:** the GPU results below used `gpu-start -- --warmup 3 --check-sentences 5 --budget 30`, because the GPU is slow.
- **adb:** `run_on_board.py` uses App Lab's copy of `adb` if `adb` is not on your PATH.
- **Where things go on the board:** everything lives under `/home/arduino/edge-bench`, and the runtimes install into folders there. Nothing in the board's system changes.
- **Network:** the benchmark runs without network access. `pip` uses the board's own network, because large downloads stalled through Docker's default network bridge on our Wi-Fi.

## Results

Measured on a 4 GB UNO Q on 2026-11-07, Debian image 20251210, with the CPU governor left at `schedutil`. The full tables, including 1-thread runs and 95th percentiles, are written to `outputs/edge/results/summary.md`.

### CPU, with ONNX Runtime

These are medians on 4 threads. "Article" means one whole MBIC article, a median of 854 words, processed sentence by sentence.

| Model | Parameters | FP32, per sentence | INT8, per sentence | INT8 speed-up | INT8, per article | INT8 peak memory | INT8 file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bert-tiny | 4.4M | 4.9 ms | 4.4 ms | 1.1× | 0.14 s | 102 MB | 4 MB |
| bert-mini | 11.1M | 24 ms | 16 ms | 1.5× | 0.50 s | 139 MB | 11 MB |
| electra-small | 13.5M | 67 ms | 44 ms | 1.5× | 1.34 s | 149 MB | 13 MB |
| minilm-l6 | 22.6M | 68 ms | 41 ms | 1.7× | 1.28 s | 185 MB | 22 MB |
| bert-small | 28.5M | 76 ms | 43 ms | 1.8× | 1.29 s | 230 MB | 27 MB |
| distilroberta | 81.5M | 239 ms | 123 ms | 1.9× | 3.69 s | 361 MB | 78 MB |
| roberta-base | 124.1M | 466 ms | 243 ms | 1.9× | 7.89 s | 417 MB | 119 MB |

1. **INT8 is faster for every model,** despite the missing dot-product instructions. On 4 threads it is 1.1 to 1.9 times faster, and the larger the model, the bigger the gain. The risk raised in the proposal does not apply to these encoders with ONNX Runtime 1.28's dynamic quantization. INT8 also cuts peak memory by up to 500 MB: RoBERTa-base drops from 915 MB to 417 MB. Whether quantization costs accuracy still has to be checked on the trained student.
2. **A student the size of MiniLM-L6 or BERT-small fits comfortably.** With INT8 on 4 threads, it handles a typical article in about 1.3 seconds, at 41 to 43 ms per sentence. Even the RoBERTa-base teacher handles an article in about 8 seconds with INT8. That is too slow to feel instant, but workable for scanning an article once when it is opened.
3. **Four threads are 1.8 to 3.4 times faster than one,** with the smallest model gaining least and the largest most. For example, MiniLM-L6 with INT8 drops from 115 ms to 41 ms per sentence.
4. **Sentence by sentence beats one padded batch.** Except for BERT-tiny, padding a whole article into one batch is slower: up to 2.3 times slower on 4 threads and 2.9 times on one thread. Padding every sentence to the article's longest sentence wastes work, so the device app should go sentence by sentence.
5. **The CPU stays cool.** The highest temperature in any run was 69.6 °C, for RoBERTa-base FP32 on 4 threads, against the first slow-down point at 90 °C. The CPU held 2.016 GHz in every run.

### Five minutes non-stop

MiniLM-L6 in INT8 on 4 threads processed articles back to back for 5 minutes, 234 articles in all. It ran at 646 words per second in the first minute and 660 in the last. The CPU peaked at 65.8 °C and never dropped below 2.016 GHz. The board does not slow down under sustained load.

### GPU, with ncnn on Vulkan

These are median times per sentence. The CPU column is ncnn on 4 threads, for a like-for-like comparison; ONNX Runtime in INT8, above, is faster still. The error columns give the largest gap from ncnn's full-FP32 CPU output on 5 sentences. For scale, the outputs themselves are at most 0.59 to 1.02.

| Model | GPU, FP16 | GPU, FP32 | CPU, ncnn, 4 threads | GPU FP32 against CPU | GPU FP16 error | GPU FP32 error |
| --- | --- | --- | --- | --- | --- | --- |
| bert-tiny | 32 ms | 40 ms | 4.0 ms | 10× slower | 0.62 | 0.0005 |
| bert-mini | 211 ms | 285 ms | 21 ms | 13× slower | 0.76 | 0.0009 |
| electra-small | 615 ms | 798 ms | 62 ms | 13× slower | 0.84 | 0.0018 |
| minilm-l6 | 718 ms | 871 ms | 60 ms | 15× slower | 0.67 | 0.0007 |
| bert-small | 797 ms | 1,023 ms | 65 ms | 16× slower | 1.14 | 0.0012 |
| distilroberta | 2,814 ms | 4,565 ms | 191 ms | 24× slower | 0.49 | 0.0011 |
| roberta-base | 5,701 ms | 8,747 ms | 381 ms | 23× slower | 0.49 | 0.0036 |

1. **FP16 on the GPU gives wrong answers.** In FP16, ncnn's default on GPUs that support it, the Adreno's output is wrong for every model: the errors are as large as the scores themselves. FP32 on the GPU gives the right answer.
2. **The GPU is much slower than the CPU, and the gap grows with model size.** In FP16 it is 8 to 15 times slower than the CPU, and in FP32 10 to 24 times slower. A MiniLM-sized student would take 30 seconds per article on the GPU, against 1.3 seconds with ONNX Runtime in INT8 on the CPU.
3. **Its one advantage is a free CPU.** During GPU runs the CPU is almost idle, at most 0.14 of one core busy, against 3.5 to 3.9 cores during CPU runs. The GPU also stays cool, at most 46 °C.
4. **Possible reasons.** The Adreno 702 is an entry-level GPU: OpenCL reports a single compute unit, and Turnip offers only Vulkan 1.0 on it. Each sentence also needs many small GPU jobs, so fixed costs may dominate. We did not profile further.

## What this means for the project

1. **The plan holds.** A distilled student the size of MiniLM-L6 or BERT-small, in INT8, handles a typical article in about 1.3 seconds on the board's CPU and needs about 200 MB.
2. **The teacher can run on the board too.** RoBERTa-base in INT8 takes about 8 seconds per article. That could serve as a slower, more accurate mode, or as a way to check the student on the device.
3. **The device app's recipe:** ship INT8, process sentence by sentence on 4 threads, and leave the GPU out unless a future driver fixes FP16.
4. **Heat is not a constraint.**

## Limits

- **Speed only.** The heads are untrained, so these numbers say nothing about accuracy. How much INT8 costs in accuracy has to be measured on the trained student.
- **No energy figures yet.** They need a USB-C power meter between the board and its power supply.
- **One setup.** The numbers come from one board and one software image: Debian 20251210, App Lab's Python image 0.5.0 and Mesa 25.2.6.
- **One GPU route.** The GPU was tested only through ncnn on Vulkan. OpenCL through Rusticl was not tried, and Qualcomm's own GPU driver is not part of this image.
