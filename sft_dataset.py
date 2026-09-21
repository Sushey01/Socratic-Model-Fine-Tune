#!/usr/bin/env python3
"""Qwen SFT helpers: assistant-only loss without flattening, or a printed response_template collator.

Phi-3 train.py does not import this. Qwen3 must keep enable_thinking=False.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from datasets import Dataset

ApplyFn = Callable[..., str]


def trl_version() -> str:
    import trl

    return getattr(trl, "__version__", "unknown")


def sft_config_accepts(name: str) -> bool:
    from trl import SFTConfig

    return name in inspect.signature(SFTConfig.__init__).parameters


def apply_qwen_chat_template(
    tokenizer,
    messages: list,
    *,
    add_generation_prompt: bool = False,
    enable_thinking: bool | None = False,
) -> str:
    """Instruct SFT string. Qwen3-Instruct: enable_thinking=False; Qwen2.5 ignores that kwarg."""
    kw: dict[str, Any] = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kw)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kw)


def wrap_tokenizer_non_thinking(tokenizer) -> None:
    """So TRL's own apply_chat_template calls stay non-thinking on Qwen3-Instruct."""
    if getattr(tokenizer, "_socratic_non_thinking_wrapped", False):
        return
    original = tokenizer.apply_chat_template

    def wrapped(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        try:
            return original(*args, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return original(*args, **kwargs)

    tokenizer.apply_chat_template = wrapped
    tokenizer._socratic_non_thinking_wrapped = True


def _sample_ending_with_assistant(dataset: Dataset) -> list:
    messages = dataset[0]["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected a non-empty 'messages' list")
    if messages[-1].get("role") == "assistant":
        return messages
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            return messages[: i + 1]
    raise ValueError("No assistant turn in sample messages; cannot extract response_template")


def extract_response_template(tokenizer, sample_messages: list, apply_fn: ApplyFn) -> str:
    """Literal assistant-turn marker from this tokenizer's chat template. Do not guess."""
    prefix = sample_messages[:-1]
    without = apply_fn(tokenizer, prefix, add_generation_prompt=False)
    with_prompt = apply_fn(tokenizer, prefix, add_generation_prompt=True)
    if with_prompt.startswith(without):
        marker = with_prompt[len(without) :]
    else:
        n = 0
        for a, b in zip(without, with_prompt):
            if a != b:
                break
            n += 1
        marker = with_prompt[n:]
    if not marker:
        raise SystemExit(
            "Could not extract a non-empty assistant response_template from apply_chat_template. "
            "Printed sample is above; set the collator marker from that string."
        )
    return marker


def flatten_messages_to_text(dataset: Dataset, tokenizer, apply_fn: ApplyFn) -> Dataset:
    def to_text(example):
        messages = example["messages"]
        if not isinstance(messages, list):
            raise ValueError("Expected 'messages' to be a list")
        return {"text": apply_fn(tokenizer, messages, add_generation_prompt=False)}

    return dataset.map(to_text, remove_columns=[c for c in dataset.column_names if c != "text"])


def prepare_qwen_sft_dataset(
    dataset: Dataset,
    tokenizer,
    *,
    enable_thinking: bool | None = False,
) -> tuple[Dataset, dict[str, Any], dict[str, Any]]:
    """Return (dataset, extra SFTConfig kwargs, extra SFTTrainer kwargs)."""
    wrap_tokenizer_non_thinking(tokenizer)

    def apply_fn(tok, messages, add_generation_prompt=False, **_kw):
        return apply_qwen_chat_template(
            tok,
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    print(f"TRL {trl_version()}", flush=True)
    if sft_config_accepts("assistant_only_loss"):
        print(
            "Using SFTConfig(assistant_only_loss=True); keeping raw 'messages' "
            "(not flattening to a text column).",
            flush=True,
        )
        extra = [c for c in dataset.column_names if c != "messages"]
        if extra:
            dataset = dataset.remove_columns(extra)
        return dataset, {"assistant_only_loss": True}, {}

    sample = _sample_ending_with_assistant(dataset)
    rendered = apply_fn(tokenizer, sample, add_generation_prompt=False)
    print("Chat template sample (tokenize=False):", flush=True)
    print(rendered, flush=True)
    marker = extract_response_template(tokenizer, sample, apply_fn)
    print(f"DataCollatorForCompletionOnlyLM response_template={marker!r}", flush=True)

    try:
        from trl import DataCollatorForCompletionOnlyLM
    except ImportError as exc:
        raise SystemExit(
            "This TRL build has no assistant_only_loss and no DataCollatorForCompletionOnlyLM. "
            "Upgrade trl or pin a version that supports completion-only SFT."
        ) from exc

    collator = DataCollatorForCompletionOnlyLM(response_template=marker, tokenizer=tokenizer)
    dataset = flatten_messages_to_text(dataset, tokenizer, apply_fn)
    print("Flattened messages → text for completion-only collator.", flush=True)
    return dataset, {"dataset_text_field": "text"}, {"data_collator": collator}
