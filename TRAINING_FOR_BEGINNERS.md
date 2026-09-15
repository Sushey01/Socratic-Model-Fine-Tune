# Training, loss, and ScienceQA — simple guide

You already know three pieces: **JSONL** (practice dialogues), **fine-tune** (the GPU job), **GGUF** (a packed file for llama.cpp / Ollama). This note is the missing middle: **what the numbers mean** and **why ScienceQA runs after each epoch**.

You do **not** need this to export GGUF. GGUF is only “save the finished tutor in a format phones/apps can load.” Loss, accuracy, and ScienceQA are how you **check whether the tutor got better**, not how you package it.

## The school analogy

| School | This project |
| --- | --- |
| Homework workbook | `socratic_train.jsonl` (and Hub dataset `socraticfinetune`) |
| Student studying that workbook every night | Fine-tune (QLoRA) on Phi-3 or Qwen |
| “How well did I copy the homework?” | **Train loss** and **token accuracy** |
| Unseen exam the teacher wrote | **ScienceQA** (256 questions the JSONL was not copied from) |
| Report card app | **Weights & Biases** |
| USB of the finished student for home | **GGUF** on Hugging Face |

Studying only the homework can make the student great at **those** pages and still weak on a **new** exam. That is why we run ScienceQA: it is a small exam the model should not have memorized.

## What happens in one `python start.py --qwen` (or Phi-3 without `--qwen`)

```text
For each epoch (you use 3):
    For many steps (batches of chat from the JSONL):
        model guesses the next tutor words
        loss = “how wrong was that guess?”
        GPU nudges the LoRA adapters a tiny bit
        every 100 steps: save checkpoint-* on disk
    END of epoch:
        pause training
        run ScienceQA on 256 held-out questions
        log acc + SRI to W&B
        continue to next epoch
After epoch 3:
    save adapters → Hugging Face model repo
Later, if you want:
    python start.py --gguf [--qwen]  → GGUF file (not part of the exam)
```

**Epoch** = one full pass over your JSONL.  
**Step** = one small batch (your logs like “step 400”). 400 steps is **not** 400 epochs.

Checkpoints (`checkpoint-400`) are **save games**. Resume loads that save. W&B is a **graph of the session**; a new W&B project does not rewind the save game.

## Train loss (the number that goes down)

The model predicts the **next token** (piece of a word) in the tutor’s reply.

**Loss** is “how surprised the model was.” Lower is better.

Example (made-up numbers, same idea as your Phi-3 run):

- Start of train: loss **2.0** — guesses are messy.
- Middle: loss **0.3**
- End: loss **~0.05** — it is very used to **your** JSONL style.

If loss falls, the model is fitting the Socratic chats. It does **not** prove it will pass ScienceQA, and it does **not** prove it will stay Socratic on new questions.

**Overfitting (simple):** loss on homework is tiny, but the exam is worse. Like memorizing the workbook answers.

## Train accuracy / mean token accuracy

This is **not** “66% ScienceQA.”

On the **training tokens**, it is: “of the tokens we asked it to predict, how many did it get exactly right?”

Example: target sentence `Let's think about voltage.`  
If it predicts most of those tokens correctly, token accuracy is high (your Phi-3 run was about **0.98** = 98% of **train** tokens).

That only means: “it learned to copy the homework language.” A student can score 98% copying notes and still dump the answer to a new MCQ. That is why we also run ScienceQA.

## Why ScienceQA after **each epoch** (not instead of the JSONL)

The JSONL **is** the fine-tune data. ScienceQA is **not** mixed into that training file. After an epoch we **stop updating weights for a bit** and ask: “on 256 **new** natural-science multiple-choice items (text only, grades 3–10), how does the tutor behave?”

Two scores:

### 1. `eval/scienceqa_acc` — exam letter

Prompt: “Answer with A, B, C, or D only.”  
We read the first letter and compare to the gold letter.

Example: 256 questions, 170 correct → **0.6641 (66%)**.  
Guessing among 4 choices is about **25%**. 66% means it knows a lot of the science **in this text-only quiz**. It is **not** the public ~90% ScienceQA number (those tests often use images / vision models).

### 2. `eval/scienceqa_sri` — Socratic restraint

Different prompt: Grade-10 tutor system message + “do not tell me which letter is correct.”  
SRI = fraction of replies that do **not** paste the gold answer text or say “the answer is B.”

Example: 229 / 256 → **0.8945 (89%)**. High SRI = fewer spoilers. It is a **simple check**, not a full “good teacher” grade.

We run this **after every epoch** so you can see in W&B:

- Epoch 1: acc 0.58, SRI 0.80  
- Epoch 2: acc 0.64, SRI 0.86  
- Epoch 3: acc 0.66, SRI 0.89  

(Those epoch-by-epoch numbers are the **idea**. Your first Phi-3 train **skipped** ScienceQA because eval crashed; you later got 66% / 89% with `--eval` on the finished adapters.)

If acc went **up** but SRI went **down**, the model might be turning into an answer-dumping exam bot. That is the reason to look at **both** numbers, not only loss.

## What is **not** measured

| You might think | Reality |
| --- | --- |
| Train loss = exam score | No. Loss is homework. ScienceQA is the quiz. |
| Token accuracy = ScienceQA acc | No. 98% tokens vs 66% letters are different tests. |
| GGUF quality = extra accuracy | No. GGUF is compression/export. You should get similar behaviour if convert worked. |
| One W&B project = one model | You now have Phi-3 vs Qwen3-4B vs Qwen2.5-7B **projects** so charts do not mix. |
| Base GGUF = ready to LoRA-train | No. Train Hugging Face weights with QLoRA, then export a **new** GGUF. |
| LoRA vs QLoRA for 7B | On a college GPU use **QLoRA** (4-bit base + LoRA adapters). fp16 LoRA on 7B usually runs out of VRAM. |

## Tiny example of the two ScienceQA modes

Question: *Ice floats on water because…*  
Choices: A density B color C magnetism D sound  
Gold: **A**.

- **Exam mode:** model should reply `A`. Count a hit if we parse `A`.
- **Tutor mode:** a good reply is *“What happens to density when water freezes?”*  
  A bad SRI reply is *“The answer is A, ice is less dense.”*

Fine-tune JSONL teaches the second style. ScienceQA checks both styles on questions **outside** that JSONL (`ngram_overlap` 0.00 on Phi-3 meant those 256 questions did not share 5-word chunks with the train file).

## Where to look

| Place | What you see |
| --- | --- |
| Terminal `[train] step=... loss=...` | Homework fit, every few steps |
| Terminal `ScienceQA epoch=... acc=... sri=...` | Quiz after each epoch |
| W&B `socratic-phi3` | Phi-3 graphs |
| W&B `science_socratic_qwen3-4b_instruct` | Qwen3-4B graphs |
| W&B `science_socratic_qwen25-7b_instruct` | Qwen2.5-7B graphs |
| Hugging Face adapters | The LoRA files the GPU learned |
| Hugging Face `gguf/` | Optional later export for apps |

More detail (filters, formulas): [BENCHMARK.md](BENCHMARK.md).
