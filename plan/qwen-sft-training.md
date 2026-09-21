# Qwen SFT training (done)

Scope: `train_qwen.py` (Qwen3-4B) and `train_qwen25.py` (Qwen2.5-7B). Shared `make_sft_config` in `train.py` still drops unknown kwargs. Phi-3 call site stays 3 epochs, constant LR, max_length 512. LoRA `r=8`, `alpha=16`, 4-bit NF4 unchanged.

## Implemented

1. **Assistant-only loss** (`sft_dataset.py`)
   - If TRL `SFTConfig` has `assistant_only_loss`: keep raw `messages`, do not flatten to `text`, pass `assistant_only_loss=True`.
   - Else: print `apply_chat_template(..., tokenize=False)`, derive the literal assistant-turn marker, use `DataCollatorForCompletionOnlyLM`.
   - Qwen3-Instruct: `enable_thinking=False`, including when TRL applies the template.
   - Train logs print `TRL <version>` and which path was used.
2. **`--epochs` default 1** (still overridable, e.g. `--epochs 2`).
3. **`lr_scheduler_type="cosine"`**, **`warmup_ratio=0.03`**.
4. **`--max-length` default 1024**.

## Coordinate with dataset

Shorten chats flagged at >700 tokens before relying on 1024, or pass `--max-length` at launch.
