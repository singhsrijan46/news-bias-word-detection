# On-board explanations: what a small language model can and cannot do

The proposal planned an on-board language model that is "used only to explain one word on request". The encoder model scans every article; the language model is far too slow for that. This document describes what was built, how it is kept honest, and how good its explanations are.

## What it does

On the board's page, select a marked word, then press **Ask the board's AI why**. The answer has two parts:
- **Facts the board measured:** how many readers in ten would flag the word, its band, and whether the paper's bias lexicon lists it. These are always shown.
- **The AI's reason, marked experimental:** one sentence, such as *Some readers might find "starkly" one-sided here because it implies a dramatic and extreme level of inequality, which could be seen as an overstatement.* Next to it, the page says how often such reasons held up in the test below.

**The model.** Qwen2.5-0.5B-Instruct, licensed Apache 2.0, in the onnx-community project's ONNX version with 4-bit weights, at a pinned version (`scripts/app/fetch_llm.py`).
- **Runtime:** ONNX Runtime, the same runtime and container as the bias-word model, so no new software runs on the board and nothing goes online.
- **Memory:** loaded on first use and unloaded after five idle minutes. With it loaded, the app's container used 1.66 GB of the board's 3.6 GB.
- **Disk:** the 793 MB model is stored once on the board and linked into the app at each install.

**Why not Gemma 3 1B,** the model the proposal named? Its licence must be accepted on Hugging Face before download, and only the board's owner can do that. Qwen2.5 needs no acceptance.

## How the answer is kept honest

1. **The facts do not come from the language model.**
2. **The model finishes a sentence that the code starts:** *Some readers might find "X" one-sided here because it…* Decoding is greedy, so the same word in the same sentence always gets the same answer.
3. **Only short reasons in the model's own words are shown:** at most 20 words, and repeating at most 3 words of the sentence in a row. In testing, longer reasons and reasons that copied the sentence were mostly confused. One of those repeated the offensive wording of its source sentence.
4. **No replacement words from this model.** Asked for a more neutral word, it often changed the meaning: "extreme" became "moderate". Asked whether two words mean the same, it said yes for that pair too, so it cannot check itself. The bias-word model can only check that a replacement is less loaded, not that it means the same. For models that pass testing, the code can offer replacements, tested with the bias-word model.
5. **The page labels the reason experimental and gives its record.**

## Choosing the model

| Model | Runtime | Result |
| --- | --- | --- |
| Qwen2.5-0.5B-Instruct, INT8 | ONNX Runtime | Poor text in early prompts, and 1.8 tokens a second on the board |
| Qwen3-0.6B, INT8 | ONNX Runtime | Slightly better reasons, much worse replacement words |
| Qwen2.5-1.5B-Instruct, INT8 | ONNX Runtime, PC | Often off topic: once a political rant instead of an explanation |
| **Qwen2.5-0.5B-Instruct, 4-bit** | **ONNX Runtime** | **Chosen.** On topic, but generic and often partly wrong. 3.1 tokens a second on the board |
| Qwen2.5-1.5B-Instruct, 4-bit | ONNX Runtime, PC | Clearly better, in the comparison below. The 0.5B model already peaked at 1.9 GB on the board, so this one, three times larger, would not fit under ONNX Runtime |
| Qwen2.5-1.5B-Instruct, GGUF Q4_K_M | llama.cpp | Not run. Running llama.cpp's downloaded release binaries on the board was held for the owner's approval |

Prompts mattered as much as models:
- **Plain instructions:** the small models latched onto words in the instructions, or wandered off topic.
- **Worked examples in the prompt:** the models copied them back.
- **Finishing a started sentence:** this worked best, and is what the board uses.

## Speed on the board

| Qwen2.5-0.5B-Instruct | INT8 | 4-bit |
| --- | --- | --- |
| Load | 12 s | 19 s |
| Reading a 92-token prompt | 4.4 s | 5.9 s |
| Writing | 1.8 tokens a second | 3.1 tokens a second |
| Peak memory | 1.6 GB | 1.9 GB |

In the app, the first explanation took 33 seconds, including loading the model. Later ones took 13 to 16 seconds.

## How good are the explanations?

**The test.** `scripts/app/evaluate_explainer.py` draws 40 words that the board's model flags in BABE's sentences that are not in MBIC, one word per sentence, with a fixed seed.
- **Same conditions:** both models get the same prompts and the same rules for what is shown.
- **Judging:** each reason shown was judged sensible, partly right, or wrong by the developer. This was one person's reading, not a reader study.

| Model | Reasons shown, of 40 | Sensible | Partly right | Wrong | Replacement words |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B, 4-bit: the board's | 13 | 4 | 4 | 5 | Not offered |
| Qwen2.5-1.5B, 4-bit: on the PC | 30 | 13 | 12 | 5 | 23 passed the bias-word check; about 15 of them were good |

**The board's model, in its own words:**
- **Sensible:** "starkly" because it *implies a dramatic and extreme level of inequality, which could be seen as an overstatement*; "vile" because it *implies a negative or harsh characterization of Donald Trump's actions*.
- **Partly right:** "lied" because it *focuses on the act of lying to them, which is a primary point of contention in the sentence*.
- **Wrong:** "advocated" because it *focuses solely on the action of advocating, while the sentence also mentions that Bolsonaro strongly advocated it*. That reason is circular.

**The 1.5B model:**
- **Sensible:** "accused" because it *implies a negative stance towards Democrats, suggesting they are guilty of something without providing evidence or context*; "weaponized" because it *implies that the Justice Department's actions were malicious or aggressive*.
- **Good replacements:** "doubt" for "skepticism", "dispute" for "feud", "unclear" for "murky".
- **Bad replacements:** some still change the facts. "Lied" became "suggested", and "lies" became "arguments".

## What this means

1. **The proposal's design holds.** A language model on this board takes 13 seconds or more for one word, so it can only explain on request. The encoder scans articles.
2. **A 0.5B model is at the edge of useful.** It gives a reason for a third of words, and about a third of those hold up. The page shows its record, so readers can weigh it.
3. **A 1.5B model would be worth it.** It gives three times as many sensible reasons. It needs llama.cpp on the board, because under ONNX Runtime it does not fit in memory. This is a decision for the board's owner.
4. **Replacement words need a meaning check.** A less loaded word is not always a fair one: "lied" and "suggested" differ in fact, not just in tone.

## Run it

```bash
python scripts/app/fetch_llm.py              # the model, 793 MB (outputs/llm)
python scripts/app/evaluate_explainer.py     # the 40-word test (outputs/explanations)
python scripts/app/deploy_app.py install     # copies the model to the board once; --no-llm leaves it out
```

## Limits

- **Small test, one judge:** 40 words, judged by one person.
- **A general model:** it was not trained on media bias, and it knows nothing of MBIC's readers. Its reasons are its own guesses about the word.
- **Answers are fixed but can be wrong:** greedy decoding gives the same answer every time, but a wrong answer stays wrong. The page says so next to every answer.
