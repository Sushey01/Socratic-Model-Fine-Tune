"""v9: two Socratic paths per intent (on-track + stuck)."""

from __future__ import annotations

from gold_v8_stems import SYSTEM


def _as_question(text: str) -> str:
    text = (text or "").strip()
    if "?" in text:
        return text
    return text.rstrip(". ") + "?"


def two_paths(s: dict[str, str], stuck: str) -> list[dict]:
    q, ask, hint, follow, wrong, right, close = (
        s["q"],
        _as_question(s["ask"]),
        _as_question(s["hint"]),
        _as_question(s["follow"]),
        s["wrong"],
        s["right"],
        s["close"],
    )
    if "?" not in ask or "?" not in hint or "?" not in follow:
        raise ValueError(f"Missing ? in stem: {q!r}")
    on_track = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": ask},
            {"role": "user", "content": right},
            {"role": "assistant", "content": follow},
            {"role": "user", "content": "Yes, that fits."},
            {"role": "assistant", "content": close},
        ]
    }
    if stuck == "wrong":
        u1 = wrong
        a1 = "Let's test that — " + hint
    elif stuck == "idk":
        u1 = "I don't know."
        a1 = hint
    else:
        u1 = "Explain it more clearly."
        a1 = hint
    stuck_chat = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": ask},
            {"role": "user", "content": u1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": right},
            {"role": "assistant", "content": follow},
            {"role": "user", "content": "Okay, I see it."},
            {"role": "assistant", "content": close},
        ]
    }
    return [on_track, stuck_chat]
