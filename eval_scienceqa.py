"""ScienceQA K-10 natural-science eval: MC accuracy + Socratic Restraint Index (SRI)."""

from __future__ import annotations

import json
import re
import string
from pathlib import Path

import torch
from datasets import load_dataset

SOCRATIC_SYSTEM = (
    "You are a Socratic Science Tutor for a Grade 10 student, covering the full Grade 10 "
    "science curriculum. Never give the final answer directly. Guide the student toward it "
    "with questions. Keep responses to 1-3 sentences."
)
SCIENCEQA_ID = "derek-thomas/ScienceQA"
EVAL_N = 256
EVAL_SEED = 42
GRADE_MIN, GRADE_MAX = 3, 10
LETTERS = string.ascii_uppercase


def _grade_num(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).lower().replace("grade", "").replace(" ", "")
    digits = re.sub(r"[^0-9]", "", text)
    return int(digits) if digits else None


def _is_natural_science(row: dict) -> bool:
    subject = str(row.get("subject") or "").strip().lower()
    return subject in {"natural science", "natural sciences", "science"}


def format_choices(choices: list) -> str:
    return "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


def exam_user_prompt(question: str, choices: list) -> str:
    return (
        "Answer this multiple-choice science question. Reply with a single letter "
        f"(A, B, C, or D) only.\n\nQuestion: {question}\n\n{format_choices(choices)}"
    )


def tutor_user_prompt(question: str, choices: list) -> str:
    return (
        f"{question}\n\nChoices:\n{format_choices(choices)}\n\n"
        "Help me reason about this. Do not tell me which letter is correct."
    )


def parse_letter(text: str, n_choices: int) -> str | None:
    cleaned = text.strip().upper()
    match = re.search(r"\b([A-Z])\b", cleaned)
    if not match:
        match = re.match(r"^\s*([A-Z])", cleaned)
    if not match:
        return None
    letter = match.group(1)
    if LETTERS.index(letter) >= n_choices:
        return None
    return letter


def sri_restrained(reply: str, gold_text: str, gold_letter: str) -> bool:
    """True if the tutor did not dump the gold answer (SRI success)."""
    blob = reply.lower()
    gold = gold_text.strip().lower()
    if gold and gold in blob:
        return False
    if re.search(rf"\b(?:answer|correct(?:\s+choice)?)\s*(?:is|:)\s*{gold_letter}\b", blob):
        return False
    if re.search(rf"\bthe\s+answer\s+is\s+{gold_letter}\b", blob):
        return False
    return True


def ngram_overlap(questions: list[str], train_jsonl: Path, n: int = 5) -> float:
    """Fraction of eval questions whose n-gram appears in the SFT jsonl."""
    if not train_jsonl.is_file() or not questions:
        return 0.0
    corpus = train_jsonl.read_text(encoding="utf-8").lower()

    def grams(text: str) -> set[str]:
        toks = re.findall(r"[a-z0-9]+", text.lower())
        if len(toks) < n:
            return {" ".join(toks)} if toks else set()
        return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}

    hits = 0
    for q in questions:
        g = grams(q)
        if g and any(gram in corpus for gram in g):
            hits += 1
    return hits / len(questions)


def load_eval_slice(n: int = EVAL_N, seed: int = EVAL_SEED) -> list[dict]:
    cache = Path(__file__).resolve().parent / ".cache" / f"scienceqa_n{n}_seed{seed}.json"
    if cache.is_file():
        rows = json.loads(cache.read_text(encoding="utf-8"))
        print(f"ScienceQA eval slice: {len(rows)} items (cached {cache.name})")
        return rows

    print(f"Downloading ScienceQA ({SCIENCEQA_ID}) test split for eval cache...", flush=True)
    raw = load_dataset(SCIENCEQA_ID, split="test")
    drop = [c for c in raw.column_names if c in {"image", "lecture", "hint", "solution"}]
    if drop:
        raw = raw.remove_columns(drop)

    def keep(row) -> bool:
        grade = _grade_num(row.get("grade"))
        if grade is None or grade < GRADE_MIN or grade > GRADE_MAX:
            return False
        if not _is_natural_science(row):
            return False
        choices = row.get("choices") or []
        if len(choices) < 2:
            return False
        try:
            answer_idx = int(row.get("answer"))
        except (TypeError, ValueError):
            return False
        if answer_idx < 0 or answer_idx >= len(choices):
            return False
        return bool(str(row.get("question") or "").strip())

    filtered = raw.filter(keep)
    rows = []
    for row in filtered:
        answer_idx = int(row["answer"])
        choices = list(row["choices"])
        rows.append(
            {
                "question": str(row["question"]).strip(),
                "choices": [str(c) for c in choices],
                "answer_idx": answer_idx,
                "gold_letter": LETTERS[answer_idx],
                "gold_text": str(choices[answer_idx]),
                "grade": _grade_num(row.get("grade")),
            }
        )
    if not rows:
        raise RuntimeError("ScienceQA filter produced 0 natural-science grade 3–10 items.")
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(rows), generator=rng).tolist()
    picked = [rows[i] for i in perm[: min(n, len(rows))]]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(picked), encoding="utf-8")
    print(f"ScienceQA eval slice: {len(picked)} items (from {len(rows)} after filter, seed={seed})")
    return picked


def _generate(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    gen = out[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def run_scienceqa_eval(
    model,
    tokenizer,
    items: list[dict],
    *,
    exam_tokens: int = 32,
    tutor_tokens: int = 128,
) -> dict:
    model.eval()
    correct = 0
    sri_ok = 0
    examples = []
    for i, item in enumerate(items):
        exam_messages = [
            {"role": "user", "content": exam_user_prompt(item["question"], item["choices"])},
        ]
        tutor_messages = [
            {"role": "system", "content": SOCRATIC_SYSTEM},
            {"role": "user", "content": tutor_user_prompt(item["question"], item["choices"])},
        ]
        exam_out = _generate(model, tokenizer, exam_messages, exam_tokens)
        tutor_out = _generate(model, tokenizer, tutor_messages, tutor_tokens)
        pred = parse_letter(exam_out, len(item["choices"]))
        hit = pred == item["gold_letter"]
        restrained = sri_restrained(tutor_out, item["gold_text"], item["gold_letter"])
        correct += int(hit)
        sri_ok += int(restrained)
        if i < 8:
            examples.append(
                {
                    "question": item["question"][:240],
                    "gold": item["gold_letter"],
                    "pred": pred or "",
                    "exam": exam_out[:200],
                    "tutor": tutor_out[:280],
                    "correct": hit,
                    "sri_ok": restrained,
                }
            )
    n = max(len(items), 1)
    return {
        "eval/scienceqa_acc": correct / n,
        "eval/scienceqa_sri": sri_ok / n,
        "eval/n": float(len(items)),
        "examples": examples,
    }
