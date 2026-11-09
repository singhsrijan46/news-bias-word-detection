# On-board explanations: what a small language model can and cannot do

The proposal planned an on-board language model that is "used only to explain one word on request". The encoder model scans every article; the language model is far too slow for that. This document describes what was built, how it is kept honest, and how good its explanations are.

## What it does

On the board's page, select a marked word, then press **Ask the board's AI why**. The answer has two parts:
- **Facts the board measured:** how many readers in ten would flag the word, its band, and whether the paper's bias lexicon lists it. These are always shown.
- **The AI's reason, marked experimental:** one sentence, such as *Some readers might find "plot" one-sided here because it implies a secretive or hidden nature, which could be seen as negative without context.* Next to it, the page says how the model did in the test below.

The page does not suggest a replacement word. In testing, a quarter of the model's suggestions changed what the sentence says.

**The model.** Qwen2.5-1.5B-Instruct, licensed Apache 2.0, in Qwen's own GGUF version with 4-bit weights (Q4_K_M), at a pinned version (`scripts/app/fetch_llm.py`).
- **Runtime:** llama.cpp's server, from the llama.cpp project's release build for 64-bit ARM Linux (release b12207, MIT licence). The fetch script checks the archive against its published SHA-256, and keeps only the server and the libraries it loads. The RPC backend, which can hand work to other machines, is left out.
- **How it runs:** inside the app's own container, started by the app on first use and stopped after five idle minutes.
  - **No network:** it listens on a Unix socket in a private folder, not on a network port, and runs offline. Its web page and slot listing are off, and so are its agent tools, by default.
  - **No text in logs:** its log is read and thrown away, so no text reaches the app's log. The log is printed only if the server fails to start, before it has seen any text.
  - **Fixed settings:** the app starts it without any `LLAMA_` environment variables, so only the app's own settings apply.
- **Memory:** 1.2 GB with the model loaded. Almost all of it is the model file, which the server maps into memory, so Linux can drop it and read it again when memory runs short.
- **Disk:** the 1.1 GB model and the 30 MB server are stored once on the board and linked into the app at each install.

A network log taken while the board explained words shows no traffic from the app other than its web server (`docs/NETWORK_LOG.md`).

**Why llama.cpp.** Under ONNX Runtime, the 0.5B model already peaked at 1.9 GB on the board, so a model three times larger would not fit. llama.cpp keeps the 4-bit weights as they are and maps them from the file.

**Why not Gemma 3 1B,** the model the proposal named? Its licence must be accepted on Hugging Face before download, and only the board's owner can do that. Qwen2.5 needs no acceptance.

**The smaller model is still there.** `fetch_llm.py --model qwen2.5-0.5b-instruct` gets the 0.5B model the board used first. It runs with the app's own ONNX Runtime, so it needs no llama.cpp, but its reasons held up less often (below).

## How the answer is kept honest

1. **The facts do not come from the language model.**
2. **The model finishes a sentence that the code starts:** *Some readers might find "X" one-sided here because it…* Decoding is greedy, so the same word in the same sentence always gets the same answer. A sentence longer than 600 characters is cut to that much around the word, for the prompt only.
3. **Only short reasons in the model's own words pass the checks:** at most 20 words, and repeating at most 3 words of the sentence in a row. In testing, longer reasons and reasons that copied the sentence were mostly confused. One of those repeated the offensive wording of its source sentence. A reason that fails is still shown, under a warning that names the check it failed, so a reader can see everything the model wrote.
4. **No replacement words.** Asked for a more neutral word, both models often changed the meaning. The 0.5B model turned "extreme" into "moderate". The 1.5B model on the board turned "kill" into "target", and "lied" into "allegedly".
   - **No way to check meaning:** the bias-word model can check that a replacement is less loaded, but not that it means the same. Its check even favours words that say less: "battleground" became "location".
   - **Still in the code:** the code can offer replacements, tested with the bias-word model, for a model whose `llm_info.json` turns them on.
5. **The page labels the reason experimental and gives its record.**

## Choosing the model

| Model | Runtime | Result |
| --- | --- | --- |
| Qwen2.5-0.5B-Instruct, INT8 | ONNX Runtime | Poor text in early prompts, and 1.8 tokens a second on the board |
| Qwen3-0.6B, INT8 | ONNX Runtime | Slightly better reasons, much worse replacement words |
| Qwen2.5-1.5B-Instruct, INT8 | ONNX Runtime, PC | Often off topic: once a political rant instead of an explanation |
| Qwen2.5-0.5B-Instruct, 4-bit | ONNX Runtime | Used first. On topic, but generic and often wrong. 3.1 tokens a second on the board |
| Qwen2.5-1.5B-Instruct, 4-bit | ONNX Runtime, PC | Better, in the comparison below, but would not fit in the board's memory under ONNX Runtime |
| **Qwen2.5-1.5B-Instruct, GGUF Q4_K_M** | **llama.cpp** | **Chosen.** About as good as under ONNX Runtime, with more than twice as many sensible reasons as the 0.5B model. 3.4 tokens a second on the board, in 1.2 GB |

Prompts mattered as much as models:
- **Plain instructions:** the small models latched onto words in the instructions, or wandered off topic.
- **Worked examples in the prompt:** the models copied them back.
- **Finishing a started sentence:** this worked best, and is what the board uses.

## Speed on the board

| | Qwen2.5-0.5B, INT8, ONNX Runtime | Qwen2.5-0.5B, 4-bit, ONNX Runtime | **Qwen2.5-1.5B, Q4_K_M, llama.cpp** |
| --- | --- | --- | --- |
| Load | 12 s | 19 s | 4 s, or about 12 s after a reboot |
| Reading the prompt | 21 tokens a second | 16 tokens a second | 5.4 tokens a second |
| Writing | 1.8 tokens a second | 3.1 tokens a second | 3.4 tokens a second |
| Peak memory | 1.6 GB | 1.9 GB | 1.2 GB, almost all of it the mapped model file |

llama.cpp writes faster than ONNX Runtime did with a model a third the size, but reads the prompt three times slower. The board's CPU (four Cortex-A53 cores) lacks the dot-product instructions that llama.cpp's faster code needs, so llama.cpp uses its basic ARMv8.0 code. Two things help:
- **A shared start:** a prompt that starts like the last one reuses the work already done, so the instructions at the start are read only once.
- **Short prompts:** the prompt is about 75 tokens for a typical sentence.

In the app, an explanation took 18 to 21 seconds. The first one after the app starts also starts the server: it took 26 seconds, or 32 seconds right after a reboot, when the model has to be read from storage. In the 40-word test, the median reason took 21 seconds.

## How good are the explanations?

**The test.** `scripts/app/evaluate_explainer.py` draws 40 words that the board's model flags in BABE's sentences that are not in MBIC, one word per sentence, with a fixed seed.
- **Same conditions:** all models get the same prompts and the same rules for what is shown.
- **On the board:** with `--board`, the language model runs on the board, in llama.cpp's server with the app's settings. The bias-word model runs on the PC, which gives the board's scores (`docs/EDGE_BENCHMARK.md`). The ONNX runs were on the PC.
- **Judging:** the developer judged every reason shown, for all the runs below in one sitting:
  - **Sensible:** says correctly what the word adds in this sentence, such as a judgement or a connotation, even if briefly.
  - **Partly right:** points at something real, but is generic ("without considering other perspectives") or adds a wrong detail.
  - **Wrong:** misreads the word or the sentence, is circular, or is cut off.

  This is one person's reading, not a reader study. An earlier, more lenient reading of the two ONNX runs gave 4, 4 and 5 for the 0.5B model, and 13, 12 and 5 for the 1.5B model.

| Model | Runtime, where | Reasons shown, of 40 | Sensible | Partly right | Wrong |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B, 4-bit: used first | ONNX Runtime, PC | 13 | 3 | 3 | 7 |
| Qwen2.5-1.5B, 4-bit | ONNX Runtime, PC | 30 | 9 | 6 | 15 |
| **Qwen2.5-1.5B, Q4_K_M: the board's** | **llama.cpp, board** | **26** | **7** | **10** | **9** |
| Qwen2.5-1.5B, Q4_K_M, no repetition penalty | llama.cpp, board | 23 | 2 | 10 | 11 |

- **Against the 0.5B model:** the board's 1.5B model gave a reason for twice as many words, and 17 sensible or partly right reasons against 6.
- **llama.cpp against ONNX Runtime:** the 1.5B model wrote much the same reasons in both, for "prestigious", "surging" and "zeroing" almost word for word. llama.cpp's run had fewer wrong reasons and fewer sensible ones. The two runs differ in the 4-bit format, and in the repetition penalty, which llama.cpp also applies to the end of the prompt.
- **The repetition penalty matters:** without it, the board's model gave 2 sensible reasons instead of 7, so the board keeps it (`repeat_penalty` in the model's `llm_info.json`). The setting was chosen on the same 40 words it is reported on.

**The board's model, in its own words:**
- **Sensible:** "shabby" because it *implies a negative view of Canada's health system without considering its strengths or other possible explanations for the poor results*; "accused" because it *implies a negative stance towards Democrats without providing equal perspective on their actions or intentions*.
- **Partly right:** "vile" because it *focuses solely on negative aspects of the action, potentially ignoring other perspectives or consequences of Trump's policies*. That is true, but it could be said of almost any loaded word, and many reasons fall back on it.
- **Wrong:** "prestigious", in a sentence about Scientific American's endorsement of Joe Biden, because it *implies that Scientific American is the only prestigious magazine endorsing Joe Biden*. The sentence says no such thing.

**Replacement words.** The test also asked for them, although the board does not show them:

| Model | Passed the bias-word check | Good | Awkward | Changed the meaning |
| --- | --- | --- | --- | --- |
| Qwen2.5-1.5B, ONNX Runtime, PC | 23 | 14 | 3 | 6 |
| Qwen2.5-1.5B, llama.cpp, board | 16 | 9 | 3 | 4 |

- **Good:** "doubt" for "skepticism", "dispute" for "feud", "poor" for "shabby", "avoiding" for "sidestepping".
- **Changed the meaning:** "target" for "kill"; "allegedly" for "lied", which gave "You allegedly to us under oath"; "location" for "battleground".
- **A milder word can be the wrong help:** "threaten" for "blackmail" kept the meaning of a sentence whose problem is not the word but its claim, an antisemitic conspiracy theory. A milder word only makes that claim read as neutral.

## What this means

1. **The proposal's design holds.** A language model on this board takes about 20 seconds for one word, so it can only explain on request. The encoder scans articles.
2. **The 1.5B model is worth running, but it is far from reliable.** Through llama.cpp it fits in 1.2 GB. It gives a reason for two thirds of the words; a quarter of those reasons are sensible, and more than a third partly right. The page shows that record next to every answer.
3. **Generic reasons are the main weakness.** Many reasons say that a word ignores "other perspectives". The filters cannot catch these, because they are short and in the model's own words.
4. **Replacement words need a meaning check.** A less loaded word is not always a fair one: "kill" and "target" differ in fact, not just in tone.

## Run it

```bash
python scripts/app/fetch_llm.py                                   # the model and llama.cpp, 1.1 GB (outputs/llm)
python scripts/app/deploy_app.py install                          # copies them to the board once; --no-llm leaves them out
python scripts/app/evaluate_explainer.py --board --suggestions    # the 40-word test (outputs/explanations)
```

## Limits

- **Small test, one judge:** 40 words, judged by one person. The repetition penalty was chosen on the same words.
- **A general model:** it was not trained on media bias, and it knows nothing of MBIC's readers. Its reasons are its own guesses about the word.
- **Answers are fixed but can be wrong:** greedy decoding gives the same answer every time, but a wrong answer stays wrong. The page says so next to every answer.
- **Outside code:** the board runs llama.cpp's release build as published, checked against its SHA-256, not a build made from source here.
