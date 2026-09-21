#!/usr/bin/env python3
"""Fine-tune Phi-3-mini on the Socratic Grade 10 chat dataset.

Checkpoints are written under OUTPUT_DIR (default ./socratic_finetuned_model) on this machine only.
Hugging Face gets the final LoRA adapters after training finishes. Re-run to resume from the latest local checkpoint.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import getpass
import json
import os
import re
import sys
import traceback
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from eval_scienceqa import EVAL_N, load_eval_slice, ngram_overlap, run_scienceqa_eval

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "socratic_train.jsonl"
DEFAULT_OUTPUT = ROOT / "socratic_finetuned_model"
DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ENV_PATH = ROOT / ".env"
_ENV_LOADED = False


def load_runtime_env() -> None:
    """Load gitignored .env so train_qwen.py does not prompt when keys already exist."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    try:
        from dotenv import load_dotenv

        if ENV_PATH.is_file():
            load_dotenv(ENV_PATH, override=False)
    except ImportError:
        if ENV_PATH.is_file():
            raw = ENV_PATH.read_text(encoding="utf-8-sig")
            for line in raw.splitlines():
                line = line.strip().lstrip("\ufeff")
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'").strip('"')
                if key and key not in os.environ:
                    os.environ[key] = val
    for name in ("WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = (os.environ.get(name) or "").strip().strip("'").strip('"')
        if val:
            os.environ[name] = val
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    os.environ.setdefault("HF_QWEN_REPO", "Susu11/Science_Socratic_Qwen3-4B_Instruct")
    os.environ.setdefault("HF_QWEN25_REPO", "Susu11/qwen2.5-7b-socratic-tutor")
    os.environ.setdefault("WANDB_PROJECT_QWEN", "science_socratic_qwen3-4b_instruct")
    os.environ.setdefault("WANDB_PROJECT_QWEN25", "science_socratic_qwen25-7b_instruct")
    if (os.environ.get("WANDB_API_KEY") or "").strip():
        print("Using WANDB_API_KEY from .env", flush=True)
    if (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip():
        print("Using HF_TOKEN from .env", flush=True)


def load_base_config(model_id: str):
    """Hub Phi-3 config often has rope_type but old modeling_phi3.py wants rope_scaling['type']."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    rs = getattr(config, "rope_scaling", None)
    if isinstance(rs, dict) and "type" not in rs:
        copied = dict(rs)
        if copied.get("rope_type"):
            copied["type"] = copied["rope_type"]
            config.rope_scaling = copied
        else:
            config.rope_scaling = None
    return config


def from_pretrained_phi3(model_id: str, **kwargs):
    """Load Phi-3; fall back to transformers built-in class if remote rope_scaling crashes."""
    kwargs.setdefault("attn_implementation", "eager")
    config = kwargs.pop("config", None) or load_base_config(model_id)
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id, config=config, trust_remote_code=True, **kwargs
        )
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Remote Phi-3 code failed ({exc!r}); loading built-in transformers Phi-3.", flush=True)
        kwargs.pop("trust_remote_code", None)
        return AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=False, **kwargs)


def load_jsonl(path: Path) -> Dataset:
    rows = []
    errors = []
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"line {i}: {exc}")
    if errors:
        print("JSONL parse errors:", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        raise SystemExit(1)
    if not rows:
        raise SystemExit(f"No examples found in {path}")
    if "messages" not in rows[0]:
        raise SystemExit(f"{path} must contain a 'messages' field (use socratic_train.jsonl)")
    print(f"Loaded {len(rows)} examples from {path}")
    return Dataset.from_list(rows)


def resolve_checkpoint(output_dir: Path, resume: bool) -> str | None:
    if not resume or not output_dir.is_dir():
        return None
    last = get_last_checkpoint(str(output_dir))
    if last:
        print(f"Resuming from checkpoint: {last}")
    else:
        print("No checkpoint found; starting a new run.")
    return last


def write_model_card(output_dir: Path, repo_id: str, base_model: str) -> None:
    card = f"""---
library_name: peft
base_model: {base_model}
tags:
  - lora
  - sft
  - socratic
  - education
---

# {repo_id}

Grade 10 Socratic science tutor LoRA. **GitHub** holds code and data; **this Hub repo** holds weights.

## What is in this repo

1. **Deploy (repo root):** final PEFT adapters + tokenizer after training finishes. Load with `PeftModel.from_pretrained("{repo_id}")` on top of `{base_model}`.
2. **Resume:** Trainer `checkpoint-*` folders stay on the training PC (`socratic_finetuned_model/`). They are not uploaded.
3. **GGUF (later):** after fine-tune, `bash start.sh --gguf` (merge + llama.cpp quant). Not uploaded during training.

Local training still snapshots every 100 steps and keeps the last 3 checkpoint folders on disk.

## Linked models

- CUDA train base: [{base_model}](https://huggingface.co/{base_model})
- Notebook / MLX 4-bit reference: [Oscilla/Phi-3.5-mini-instruct-mlx-4Bit](https://huggingface.co/Oscilla/Phi-3.5-mini-instruct-mlx-4Bit)
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def _hub_checkpoint_names(api, repo_id: str) -> list[str]:
    names: set[str] = set()
    try:
        for item in api.list_repo_tree(repo_id, repo_type="model", recursive=False):
            path = getattr(item, "path", None) or str(item)
            if path.startswith("checkpoint-"):
                names.add(path.split("/")[0])
    except Exception as exc:
        print(f"Could not list Hub checkpoints ({exc}); skip prune.")
    return sorted(names)


def push_output(output_dir: Path, repo_id: str, base_model: str, write_card=None) -> None:
    from huggingface_hub import HfApi, login

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, repo_type="model", private=True)

    (write_card or write_model_card)(output_dir, repo_id, base_model)

    print(f"Uploading final adapters + tokenizer (no checkpoints) to {repo_id}")
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=[
            "checkpoint-*",
            "checkpoint-*/**",
            "runs/**",
            "tmp_trainer/**",
            "*.tmp",
        ],
    )
    for name in _hub_checkpoint_names(api, repo_id):
        print(f"Removing Hub checkpoint leftover {name} (checkpoints stay local only)")
        api.delete_folder(repo_id=repo_id, path_in_repo=name, repo_type="model")

    print(f"Pushed final model to https://huggingface.co/{repo_id}")


def prompt_secret(env_name: str, prompt: str) -> str:
    current = os.environ.get(env_name, "").strip()
    if current:
        return current
    if not sys.stdin.isatty():
        return ""
    value = getpass.getpass(prompt).strip()
    if value:
        os.environ[env_name] = value
    return value


def prompt_line(env_name: str, prompt: str, default: str = "") -> str:
    current = os.environ.get(env_name, "").strip()
    if current and current != "YOUR_HF_USER/socratic-phi3":
        return current
    if not sys.stdin.isatty():
        return current or default
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    value = value or default
    if value:
        os.environ[env_name] = value
    return value


def ensure_runtime_secrets() -> None:
    """Ask on the training PC if W&B or Hugging Face credentials are missing."""
    load_runtime_env()
    prompt_secret(
        "WANDB_API_KEY",
        "Weights & Biases API key (wandb.ai/authorize, input hidden): ",
    )
    prompt_secret(
        "HF_TOKEN",
        "Hugging Face write token (huggingface.co/settings/tokens, input hidden): ",
    )
    if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")
    prompt_line(
        "HF_HUB_REPO",
        "Hugging Face model repo to push adapters",
        default=os.environ.get("HF_HUB_REPO") or "Susu11/socratic-phi3",
    )


def wandb_enabled() -> bool:
    return bool(os.environ.get("WANDB_API_KEY", "").strip())


def init_wandb() -> bool:
    """Start W&B using WANDB_API_KEY. Do not call wandb.login() — it hangs on Windows."""
    if not wandb_enabled():
        print("WANDB_API_KEY still missing; metrics print in the terminal only.", flush=True)
        return False
    import wandb

    os.environ.setdefault("WANDB_START_METHOD", "thread")
    os.environ.setdefault("WANDB_DISABLE_GIT", "true")
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")
    print("Connecting to Weights & Biases (90s timeout)...", flush=True)
    init_kw = dict(
        project=os.environ.get("WANDB_PROJECT", "socratic-phi3"),
        name=os.environ.get("WANDB_RUN_NAME") or None,
        config={
            "base_model": os.environ.get("BASE_MODEL", DEFAULT_MODEL),
            "eval_n": EVAL_N,
            "benchmark": "ScienceQA natural science grades 3-10",
        },
    )
    try:
        init_kw["settings"] = wandb.Settings(init_timeout=60)
    except TypeError:
        pass

    def _online():
        wandb.init(**init_kw)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_online).result(timeout=90)
    except Exception as exc:
        print(f"W&B online init failed or timed out ({exc}); offline mode.", flush=True)
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project=os.environ.get("WANDB_PROJECT", "socratic-phi3"), mode="offline")
    print(f"W&B run: {wandb.run.url if wandb.run else '(offline)'}", flush=True)
    return wandb.run is not None


def log_scienceqa_to_wandb(metrics: dict, epoch: float, step: int) -> None:
    examples = metrics.get("examples") or []
    payload = {
        "eval/scienceqa_acc": metrics["eval/scienceqa_acc"],
        "eval/scienceqa_sri": metrics["eval/scienceqa_sri"],
        "eval/n": metrics["eval/n"],
        "epoch": epoch,
    }
    if wandb_enabled():
        import wandb

        if wandb.run is not None:
            if examples:
                payload["eval/scienceqa_examples"] = wandb.Table(
                    columns=list(examples[0].keys()),
                    data=[list(row.values()) for row in examples],
                )
            wandb.log(payload, step=step)
            wandb.run.summary["eval/scienceqa_acc"] = metrics["eval/scienceqa_acc"]
            wandb.run.summary["eval/scienceqa_sri"] = metrics["eval/scienceqa_sri"]


class ScienceQAEpochCallback(TrainerCallback):
    def __init__(self, tokenizer, train_jsonl: Path):
        self.tokenizer = tokenizer
        self.train_jsonl = train_jsonl
        self.items: list | None = None

    def _ensure_items(self):
        if self.items is None:
            self.items = load_eval_slice()

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            self._ensure_items()
            overlap = ngram_overlap([x["question"] for x in self.items], self.train_jsonl)
            print(f"ScienceQA vs train jsonl 5-gram overlap: {overlap:.4f}")
            if wandb_enabled():
                import wandb

                if wandb.run is not None:
                    wandb.summary["eval/ngram_overlap"] = overlap
                    wandb.summary["eval/n"] = len(self.items)
        except Exception as exc:
            traceback.print_exc()
            print(f"ScienceQA eval setup failed ({exc}); skipping per-epoch benchmark.")
            self.items = []
        return control

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if not self.items or model is None:
            return control
        print(f"Running ScienceQA eval after epoch {state.epoch:.0f} ({len(self.items)} items)...")
        try:
            metrics = run_scienceqa_eval(model, self.tokenizer, self.items)
        except Exception as exc:
            traceback.print_exc()
            print(f"ScienceQA eval failed: {exc}")
            return control
        epoch = float(state.epoch)
        print(
            f"ScienceQA epoch={epoch:.0f} acc={metrics['eval/scienceqa_acc']:.4f} "
            f"sri={metrics['eval/scienceqa_sri']:.4f} n={int(metrics['eval/n'])}"
        )
        log_scienceqa_to_wandb(metrics, epoch, state.global_step)
        model.train()
        return control


class HeartbeatCallback(TrainerCallback):
    """Print progress even when W&B never opened a cloud run."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            bits = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in logs.items())
            print(f"[train] step={state.global_step} epoch={float(state.epoch or 0):.2f} {bits}", flush=True)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step and state.global_step % 25 == 0:
            print(f"[train] heartbeat step={state.global_step} epoch={float(state.epoch or 0):.2f}", flush=True)
        return control


def make_sft_config(**kwargs) -> SFTConfig:
    """Drop kwargs this installed TRL SFTConfig does not accept (e.g. save_safetensors)."""
    dropped: list[str] = []
    while True:
        try:
            cfg = SFTConfig(**kwargs)
            if dropped:
                print(f"SFTConfig ignored unsupported args: {', '.join(dropped)}")
            return cfg
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            if not match or match.group(1) not in kwargs:
                raise
            key = match.group(1)
            dropped.append(key)
            kwargs.pop(key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Socratic Phi-3 LoRA fine-tune")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=os.environ.get("BASE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints and start over",
    )
    parser.add_argument(
        "--push-to-hub",
        default=os.environ.get("HF_HUB_REPO", ""),
        help="HF repo id (or set HF_HUB_REPO). Uploads the final LoRA adapters only; checkpoints stay local.",
    )
    return parser.parse_args()


def main() -> None:
    load_runtime_env()
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ensure_runtime_secrets()
    if not args.push_to_hub:
        args.push_to_hub = os.environ.get("HF_HUB_REPO", "")

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU not found. This training job needs an NVIDIA GPU. "
            "On the college PC, check `nvidia-smi` and install a CUDA build of PyTorch."
        )

    dataset = load_jsonl(args.data)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bnb = True
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    try:
        model = from_pretrained_phi3(args.model, quantization_config=bnb, device_map={"": 0})
    except Exception as exc:
        print(f"4-bit bitsandbytes load failed ({exc}); retrying fp16 without 4-bit.")
        use_bnb = False
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        try:
            model = from_pretrained_phi3(args.model, dtype=dtype, device_map={"": 0})
        except TypeError:
            model = from_pretrained_phi3(args.model, torch_dtype=dtype, device_map={"": 0})

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_arguments = make_sft_config(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_8bit" if use_bnb else "adamw_torch",
        logging_steps=10,
        learning_rate=2e-4,
        fp16=not use_bf16,
        bf16=use_bf16,
        max_grad_norm=0.3,
        max_steps=-1,
        lr_scheduler_type="constant",
        report_to="wandb" if wandb_enabled() else "none",
        logging_first_step=True,
        max_length=args.max_length,
        packing=False,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        dataset_text_field="text",
    )

    def to_text(example):
        messages = example["messages"]
        if not isinstance(messages, list):
            raise ValueError("Expected 'messages' to be a list")
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    dataset = dataset.map(to_text, remove_columns=[c for c in dataset.column_names if c != "text"])
    print("Dataset mapped. Connecting W&B, then building trainer.", flush=True)

    if not init_wandb():
        training_arguments.report_to = ["none"]

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": lora_config,
        "args": training_arguments,
        "callbacks": [HeartbeatCallback(), ScienceQAEpochCallback(tokenizer, args.data)],
    }
    print("Building SFTTrainer...", flush=True)
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    resume_from = resolve_checkpoint(args.output_dir, resume=not args.no_resume)
    print("Calling trainer.train() — watch [train] heartbeat lines and GPU use.", flush=True)
    trainer.train(resume_from_checkpoint=resume_from)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapters and tokenizer to {args.output_dir}")
    last = get_last_checkpoint(str(args.output_dir))
    if last:
        print(f"Latest resume checkpoint: {last}")

    repo_id = (args.push_to_hub or os.environ.get("HF_HUB_REPO") or "").strip()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if repo_id and not token:
        token = prompt_secret(
            "HF_TOKEN",
            "Hugging Face write token required to push the model (input hidden): ",
        )
    if repo_id and token:
        os.environ["HF_TOKEN"] = token
        push_output(args.output_dir, repo_id, args.model)
    elif repo_id:
        print("Skipping Hub upload: no HF_TOKEN after prompt.")
    else:
        print("Skipping Hub upload (set HF_HUB_REPO or --push-to-hub).")

    if wandb_enabled():
        import wandb

        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
