"""
End-to-End Fine-Tuning Script for Socratic Science Tutor
Optimized for Kaggle Dual NVIDIA T4 GPUs (2x 16GB VRAM)

Configured for:
- Dataset: Susu11/socratic_idea_expansion (Hugging Face Hub or local socratic_dataset.jsonl)
- Base Model: Qwen/Qwen2.5-7B-Instruct (Recommended for Socratic conversational steering)
- Target Model Hub ID: Susu11/socratic_qwen8b
- QLoRA (4-bit NF4) + PEFT for efficient dual-GPU training
- Automatic multi-GPU balance with device_map="auto"
- Weights & Biases (wandb) experiment tracking
- Automatic Hugging Face Hub upload (LoRA adapter & tokenizer)
"""

import os
import sys
import json
import argparse
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import SFTTrainer

# Kaggle Secrets handling
try:
    from kaggle_secrets import UserSecretsClient
    IN_KAGGLE = True
except ImportError:
    IN_KAGGLE = False


def setup_auth(hf_token: str = None, wandb_key: str = None):
    """Handles Hugging Face and Wandb authentication from args, env, or Kaggle Secrets."""
    # Hugging Face Token
    if not hf_token:
        if IN_KAGGLE:
            try:
                user_secrets = UserSecretsClient()
                hf_token = user_secrets.get_secret("HF_TOKEN")
            except Exception:
                hf_token = os.getenv("HF_TOKEN")
        else:
            hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=True)
        print("✅ Successfully logged in to Hugging Face Hub.")
    else:
        print("⚠️ Warning: No HF_TOKEN provided. Dataset download/upload or model push may fail if repos are private.")

    # Wandb API Key
    if not wandb_key:
        if IN_KAGGLE:
            try:
                user_secrets = UserSecretsClient()
                wandb_key = user_secrets.get_secret("WANDB_API_KEY")
            except Exception:
                wandb_key = os.getenv("WANDB_API_KEY")
        else:
            wandb_key = os.getenv("WANDB_API_KEY")

    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
        import wandb
        wandb.login(key=wandb_key)
        print("✅ Successfully logged in to Weights & Biases.")
    else:
        print("ℹ️ No WANDB_API_KEY detected. Wandb will run in offline/disabled mode.")


def load_socratic_dataset(dataset_identifier: str):
    """
    Loads dataset either from Hugging Face Hub (e.g. Susu11/socratic_idea_expansion)
    or from a local JSONL file (e.g. socratic_dataset.jsonl).
    """
    if os.path.exists(dataset_identifier):
        print(f"📂 Loading dataset from local file: {dataset_identifier}")
        records = []
        with open(dataset_identifier, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "messages" in data and isinstance(data["messages"], list):
                        records.append({"messages": data["messages"]})
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping invalid JSON at line {line_num}: {e}")
        print(f"✅ Loaded {len(records)} verified conversation samples from local file.")
        return Dataset.from_list(records)

    print(f"🌐 Fetching dataset from Hugging Face Hub: {dataset_identifier}")
    try:
        hf_ds = load_dataset(dataset_identifier, split="train")
        print(f"✅ Loaded {len(hf_ds)} samples from Hugging Face Hub.")
        return hf_ds
    except Exception as e:
        fallback = "socratic_dataset.jsonl"
        if os.path.exists(fallback):
            print(f"⚠️ Failed to load '{dataset_identifier}' from Hub ({e}). Falling back to local '{fallback}'...")
            return load_socratic_dataset(fallback)
        raise RuntimeError(f"Could not load dataset from '{dataset_identifier}' or local file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Socratic LLM on Dual T4 GPUs")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID (Default: Qwen/Qwen2.5-7B-Instruct or meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--dataset_name", type=str, default="Susu11/socratic_idea_expansion",
                        help="Hugging Face dataset repository ID or local jsonl path")
    parser.add_argument("--output_dir", type=str, default="./socratic_tutor_lora",
                        help="Directory to save checkpoint outputs")
    parser.add_argument("--hub_model_id", type=str, default="Susu11/socratic_qwen8b",
                        help="Hugging Face repo to push model (e.g., 'Susu11/socratic_qwen8b')")
    parser.add_argument("--wandb_project", type=str, default="socratic-model-fine-tune",
                        help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default="kaggle-dual-t4-qwen-run",
                        help="Wandb run name")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (effective batch size = 2 * 2 * 4 = 16)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Initial learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    parser.add_argument("--hf_token", type=str, default="",
                        help="Hugging Face API token")
    parser.add_argument("--wandb_key", type=str, default="",
                        help="Weights & Biases API Key")
    parser.add_argument("--upload_dataset_to_hub", action="store_true",
                        help="Upload local dataset to Hugging Face Hub before training")
    parser.add_argument("--merge_and_push", action="store_true",
                        help="Merge LoRA weights with base model and push full 16-bit model to Hub")

    args = parser.parse_args()

    # 1. Hardware Detection
    gpu_count = torch.cuda.device_count()
    print(f"🖥️ Detected {gpu_count} GPU(s):")
    for i in range(gpu_count):
        print(f"   [{i}]: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB)")

    # 2. Authentication setup
    setup_auth(args.hf_token, args.wandb_key)

    # Configure Wandb environment
    if os.getenv("WANDB_API_KEY"):
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_WATCH"] = "gradients"
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # Optional: Upload dataset to Hub if requested and local file exists
    if args.upload_dataset_to_hub:
        local_file = "socratic_dataset.jsonl"
        if os.path.exists(local_file):
            print(f"📤 Uploading local '{local_file}' to Hugging Face Dataset: {args.dataset_name}...")
            local_ds = load_socratic_dataset(local_file)
            local_ds.push_to_hub(args.dataset_name, private=False)
            print("✅ Dataset successfully pushed to Hugging Face Hub!")

    # 3. Load Dataset
    raw_dataset = load_socratic_dataset(args.dataset_name)

    # 4. Tokenizer Configuration
    print(f"🔤 Loading tokenizer for: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. Format Dataset using Chat Template
    def apply_chat_template(batch):
        formatted_texts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in batch["messages"]
        ]
        return {"text": formatted_texts}

    formatted_dataset = raw_dataset.map(apply_chat_template, batched=True)
    print("📋 Sample formatted entry:")
    print(formatted_dataset[0]["text"][:400] + "\n...[truncated]...\n")

    # 6. Quantization Config (4-bit NF4 optimized for T4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # T4 optimal
        bnb_4bit_use_double_quant=True,
    )

    # 7. Model Loading with multi-GPU distribution
    print(f"🤖 Loading base model '{args.base_model}' across available GPUs...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # 8. LoRA Configuration
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
    print("📊 Trainable parameters:")
    model.print_trainable_parameters()

    # 9. Training Arguments (T4-optimized fp16)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,       # Crucial: T4 does not support BF16
        bf16=False,
        max_grad_norm=0.3,
        optim="paged_adamw_8bit",
        report_to=["wandb"] if os.getenv("WANDB_API_KEY") else ["none"],
        run_name=args.wandb_run_name,
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id if args.hub_model_id else None,
        hub_strategy="checkpoint",
        remove_unused_columns=False,
    )

    # 10. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 11. Start Training
    print("🚀 Starting SFT fine-tuning run...")
    train_result = trainer.train()
    print("✅ Training completed successfully!")
    print(f"📉 Training loss: {train_result.training_loss:.4f}")

    # 12. Save Local Artifacts
    print(f"💾 Saving LoRA adapter and tokenizer to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 13. Push to Hugging Face Hub (Adapter)
    if args.hub_model_id:
        print(f"📤 Pushing LoRA adapter to Hugging Face Hub: {args.hub_model_id}...")
        trainer.model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        print("🎉 LoRA adapter successfully pushed to Hub!")

    # 14. Optional: Merge Adapter into Base Model (Full 16-bit) and Push
    if args.merge_and_push and args.hub_model_id:
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

        merged_hub_id = f"{args.hub_model_id}-merged"
        print(f"📤 Pushing full merged 16-bit model to: {merged_hub_id}...")
        merged_model.push_to_hub(merged_hub_id)
        tokenizer.push_to_hub(merged_hub_id)
        print("🎉 Full merged model successfully pushed to Hub!")

    print("\n🏁 All steps completed successfully!")


if __name__ == "__main__":
    main()
