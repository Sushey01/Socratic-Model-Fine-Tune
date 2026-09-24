"""
End-to-End Fine-Tuning Script for Socratic Science Tutor
Optimized for Kaggle Dual NVIDIA T4 GPUs (2x 16GB VRAM)

Adaptive parameter inspection to guarantee 100% compatibility across all TRL versions.
Explicit max_memory mapping to prevent any CPU offloading ValueError.
"""

import os
import sys
import gc
import json
import argparse
import subprocess
import inspect
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfApi

# Kaggle Secrets handling
try:
    from kaggle_secrets import UserSecretsClient
    IN_KAGGLE = True
except ImportError:
    IN_KAGGLE = False


def clean_token(val: str) -> str:
    if not val:
        return None
    val = val.strip().strip("'").strip('"').strip()
    return val if val else None


def get_hf_token(fallback: str = None) -> str:
    if fallback:
        cleaned = clean_token(fallback)
        if cleaned:
            return cleaned

    for env_key in ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_KEY", "hf_token"]:
        cleaned = clean_token(os.getenv(env_key))
        if cleaned:
            return cleaned

    if IN_KAGGLE:
        try:
            secrets = UserSecretsClient()
            for label in ["HF_TOKEN", "hf_token", "HUGGINGFACE_TOKEN", "huggingface", "HF", "HF_API_KEY"]:
                try:
                    cleaned = clean_token(secrets.get_secret(label))
                    if cleaned:
                        return cleaned
                except Exception:
                    pass
        except Exception:
            pass

    return None


def get_wandb_token(fallback: str = None) -> str:
    if fallback:
        cleaned = clean_token(fallback)
        if cleaned:
            return cleaned

    for env_key in ["WANDB_TOKEN", "WANDB_API_KEY", "wandb_token", "wandb_key"]:
        cleaned = clean_token(os.getenv(env_key))
        if cleaned:
            return cleaned

    if IN_KAGGLE:
        try:
            secrets = UserSecretsClient()
            for label in ["WANDB_TOKEN", "WANDB_API_KEY", "wandb_token", "wandb_api_key", "WANDB_KEY", "wandb"]:
                try:
                    cleaned = clean_token(secrets.get_secret(label))
                    if cleaned:
                        return cleaned
                except Exception:
                    pass
        except Exception:
            pass

    return None


def setup_auth(hf_token_arg: str = None, wandb_token_arg: str = None):
    hf_token = get_hf_token(hf_token_arg)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        from huggingface_hub import login
        try:
            login(token=hf_token, add_to_git_credential=True)
            print("✅ Successfully authenticated with Hugging Face Hub.")
        except Exception as e:
            print(f"⚠️ Hugging Face Login notice: {e}")
    else:
        print("⚠️ Warning: No HF_TOKEN detected in Kaggle Secrets.")

    wandb_token = get_wandb_token(wandb_token_arg)
    if wandb_token:
        os.environ["WANDB_API_KEY"] = wandb_token
        os.environ["WANDB_PROJECT"] = "socratic-model-fine-tune"
        import wandb
        try:
            wandb.login(key=wandb_token)
            print("✅ Successfully authenticated with Weights & Biases.")
        except Exception as e:
            print(f"⚠️ Wandb login failed: {e}. Running in disabled mode.")
            os.environ["WANDB_DISABLED"] = "true"
    else:
        print("ℹ️ No WANDB_TOKEN detected. Running without Wandb tracking.")
        os.environ["WANDB_DISABLED"] = "true"


def load_socratic_dataset(hub_dataset_id: str):
    candidate_paths = [
        "/kaggle/input/datasets/shekhu1/socratic-final/socratic_dataset.jsonl",
        "/kaggle/input/socratic-final/socratic_dataset.jsonl",
        "socratic_dataset.jsonl"
    ]
    
    target_file = None
    for p in candidate_paths:
        if os.path.exists(p):
            target_file = p
            break

    if target_file:
        print(f"📂 Found local dataset at: {target_file}")
        records = [json.loads(line) for line in open(target_file, "r", encoding="utf-8") if line.strip()]
        ds = Dataset.from_list(records)
        print(f"✅ Loaded {len(ds)} samples from disk.")

        if os.getenv("HF_TOKEN"):
            try:
                print(f"📤 Syncing to Hugging Face Hub: {hub_dataset_id}...")
                ds.push_to_hub(hub_dataset_id, private=False)
                print(f"✅ Synced to: https://huggingface.co/datasets/{hub_dataset_id}")
            except Exception as e:
                print(f"ℹ️ Hub sync notice: {e}")
        return ds

    print(f"🌐 Fetching dataset from Hugging Face Hub: {hub_dataset_id}...")
    return load_dataset(hub_dataset_id, split="train")


def convert_and_upload_gguf(merged_model_dir: str, hub_model_id: str, out_gguf_name: str = "socratic_qwen_f16.gguf"):
    print("\n📦 Starting GGUF Conversion...")
    try:
        if not os.path.exists("llama.cpp"):
            print("Cloning llama.cpp for GGUF conversion...")
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git"], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "llama.cpp/requirements.txt"], check=True)
        
        print(f"Converting '{merged_model_dir}' to '{out_gguf_name}'...")
        convert_script = "llama.cpp/convert_hf_to_gguf.py"
        cmd = [
            sys.executable,
            convert_script,
            merged_model_dir,
            "--outfile", out_gguf_name,
            "--outtype", "f16"
        ]
        subprocess.run(cmd, check=True)
        print(f"✅ GGUF file created successfully: {out_gguf_name} ({os.path.getsize(out_gguf_name) / 1e9:.2f} GB)")

        if os.getenv("HF_TOKEN"):
            print(f"📤 Uploading GGUF file to Hugging Face: {hub_model_id}...")
            api = HfApi()
            api.upload_file(
                path_or_fileobj=out_gguf_name,
                path_in_repo=out_gguf_name,
                repo_id=hub_model_id,
                repo_type="model",
            )
            print(f"🎉 GGUF model successfully uploaded to: https://huggingface.co/{hub_model_id}")
    except Exception as e:
        print(f"⚠️ GGUF conversion/upload encountered an issue: {e}")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Fine-Tuning for Socratic Tutor on Dual T4 GPUs")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="Susu11/socratic_idea_expansion")
    parser.add_argument("--output_dir", type=str, default="./socratic_tutor_lora")
    parser.add_argument("--merged_dir", type=str, default="./socratic_tutor_merged")
    parser.add_argument("--hub_model_id", type=str, default="Susu11/socratic_qwen8b")
    parser.add_argument("--wandb_project", type=str, default="socratic-model-fine-tune")
    parser.add_argument("--wandb_run_name", type=str, default="kaggle-dual-t4-qwen-run")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--wandb_token", type=str, default="")
    parser.add_argument("--merge_and_push", action="store_true", default=True)
    parser.add_argument("--convert_gguf", action="store_true", default=True)

    args = parser.parse_args()

    gpu_count = torch.cuda.device_count()
    print(f"🖥️ Detected {gpu_count} GPU(s):")
    for i in range(gpu_count):
        print(f"   [{i}]: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB)")

    # Clean cache
    gc.collect()
    torch.cuda.empty_cache()

    setup_auth(args.hf_token, args.wandb_token)

    raw_dataset = load_socratic_dataset(args.dataset_name)

    print(f"🔤 Loading tokenizer for: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def apply_chat_template(batch):
        formatted_texts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in batch["messages"]
        ]
        return {"text": formatted_texts}

    formatted_dataset = raw_dataset.map(apply_chat_template, batched=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Restrict memory strictly to GPUs without CPU offload
    max_memory = {i: "14GiB" for i in range(gpu_count)}
    print(f"Target GPU Memory Allocation: {max_memory}")

    print(f"🤖 Loading base model '{args.base_model}' across available GPUs...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    base_args = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "warmup_steps": 10,
        "lr_scheduler_type": "cosine",
        "logging_steps": 5,
        "save_strategy": "epoch",
        "push_to_hub": bool(args.hub_model_id and os.getenv("HF_TOKEN")),
        "hub_model_id": args.hub_model_id if args.hub_model_id else None,
        "hub_strategy": "every_save",
        "fp16": True,
        "bf16": False,
        "max_grad_norm": 0.3,
        "optim": "paged_adamw_8bit",
        "report_to": ["wandb"] if os.getenv("WANDB_API_KEY") else ["none"],
        "run_name": args.wandb_run_name,
    }

    sft_config_params = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in sft_config_params:
        base_args["max_length"] = args.max_seq_length
    elif "max_seq_length" in sft_config_params:
        base_args["max_seq_length"] = args.max_seq_length

    if "dataset_text_field" in sft_config_params:
        base_args["dataset_text_field"] = "text"

    valid_config_args = {k: v for k, v in base_args.items() if k in sft_config_params}
    training_args = SFTConfig(**valid_config_args)

    sft_trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "train_dataset": formatted_dataset,
        "peft_config": peft_config,
        "args": training_args,
    }

    if "processing_class" in sft_trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sft_trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in sft_trainer_params and "dataset_text_field" not in valid_config_args:
        trainer_kwargs["dataset_text_field"] = "text"

    if "max_seq_length" in sft_trainer_params and "max_seq_length" not in valid_config_args and "max_length" not in valid_config_args:
        trainer_kwargs["max_seq_length"] = args.max_seq_length

    trainer = SFTTrainer(**trainer_kwargs)

    print("🚀 Starting SFT fine-tuning run (uploading to HF after each epoch)...")
    train_result = trainer.train()
    print("✅ Training completed successfully!")
    print(f"📉 Final Training Loss: {train_result.training_loss:.4f}")

    print(f"💾 Saving final LoRA adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.hub_model_id and os.getenv("HF_TOKEN"):
        print(f"📤 Pushing final adapter to Hugging Face: {args.hub_model_id}...")
        try:
            trainer.model.push_to_hub(args.hub_model_id)
            tokenizer.push_to_hub(args.hub_model_id)
            print(f"🎉 Final adapter pushed to: https://huggingface.co/{args.hub_model_id}")
        except Exception as e:
            print(f"⚠️ Error pushing final adapter: {e}")

    # Merge into 16-bit
    if args.merge_and_push:
        print("🔄 Merging LoRA adapter into base model (16-bit)...")
        del model
        del trainer
        torch.cuda.empty_cache()

        base_model_reload = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        merged_model = PeftModel.from_pretrained(base_model_reload, args.output_dir)
        merged_model = merged_model.merge_and_unload()

        os.makedirs(args.merged_dir, exist_ok=True)
        merged_model.save_pretrained(args.merged_dir)
        tokenizer.save_pretrained(args.merged_dir)
        print(f"💾 Merged model saved to: {args.merged_dir}")

        if args.hub_model_id and os.getenv("HF_TOKEN"):
            merged_hub_id = f"{args.hub_model_id}-merged"
            print(f"📤 Pushing merged 16-bit model to: {merged_hub_id}...")
            try:
                merged_model.push_to_hub(merged_hub_id)
                tokenizer.push_to_hub(merged_hub_id)
                print(f"🎉 Full merged model pushed to: https://huggingface.co/{merged_hub_id}")
            except Exception as e:
                print(f"⚠️ Error pushing merged model: {e}")

        # GGUF Conversion
        if args.convert_gguf:
            convert_and_upload_gguf(
                merged_model_dir=args.merged_dir,
                hub_model_id=args.hub_model_id,
                out_gguf_name="socratic_qwen_f16.gguf"
            )

    print("\n🏁 All training, merging, per-epoch sync, and GGUF exports completed!")


if __name__ == "__main__":
    main()
