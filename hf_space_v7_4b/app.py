import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import spaces  # noqa: E402 — must precede torch

import threading  # noqa: E402

import gradio as gr  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from peft import PeftConfig, get_peft_model, set_peft_model_state_dict  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_ID = "Susu11/v7_4b_qwen"

SYSTEM = (
    "You are a Socratic Science Tutor for a Grade 10 student, covering the full Grade 10 "
    "science curriculum. Never give the final answer directly. Guide the student toward it "
    "with questions. Keep responses to 1-3 sentences."
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True,
)
# ZeroGPU has no real CUDA at import time; PEFT's default load targets CUDA.
peft_cfg = PeftConfig.from_pretrained(ADAPTER_ID)
model = get_peft_model(model, peft_cfg)
weight_path = hf_hub_download(ADAPTER_ID, "adapter_model.safetensors")
set_peft_model_state_dict(model, load_file(weight_path, device="cpu"))
model = model.merge_and_unload()
model.eval().to("cuda")


def _chat_duration(
    message, history=None, max_new_tokens=128, *args, **kwargs
) -> int:
    """Reserve GPU seconds for one Socratic reply."""
    return int(min(90, 15 + float(max_new_tokens) * 0.04))


def _messages(message: str, history: list) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = [{"role": "system", "content": SYSTEM}]
    for turn in history or []:
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant"):
            content = turn.get("content") or ""
            if content:
                turns.append({"role": turn["role"], "content": content})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            user, assistant = turn
            if user:
                turns.append({"role": "user", "content": str(user)})
            if assistant:
                turns.append({"role": "assistant", "content": str(assistant)})
    turns.append({"role": "user", "content": message})
    return turns


@spaces.GPU(duration=_chat_duration)
def chat(message: str, history: list, max_new_tokens: int = 128):
    """Reply as the Grade 10 Socratic science tutor (Qwen3-4B QLoRA).

    Args:
        message: Student's science question or follow-up.
        history: Prior chat turns as role/content dicts.
        max_new_tokens: Cap on generated tokens (keep small for short Socratic turns).

    Yields:
        The tutor reply, streamed as it is generated.
    """
    messages = _messages(message, history)
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    if hasattr(encoded, "items"):
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in encoded.items()}
    else:
        inputs = {"input_ids": encoded.to(model.device)}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = threading.Thread(
        target=model.generate,
        kwargs=dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        ),
    )
    thread.start()
    text = ""
    for chunk in streamer:
        text += chunk
        yield text.strip()
    thread.join()


demo = gr.ChatInterface(
    fn=chat,
    title="v7 4B Socratic Grade 10 science tutor",
    description=(
        "LoRA adapters: [Susu11/v7_4b_qwen](https://huggingface.co/Susu11/v7_4b_qwen) "
        "on [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). "
        "Data: [Susu11/v7_socratic_data](https://huggingface.co/datasets/Susu11/v7_socratic_data). "
        "The tutor should **question**, not give the final answer. ZeroGPU."
    ),
    additional_inputs=[
        gr.Slider(32, 256, value=128, step=16, label="Max new tokens"),
    ],
    additional_inputs_accordion=gr.Accordion("Generation", open=False),
    examples=[
        ["Why does ice float on water?"],
        ["What is force?"],
        ["What is energy?"],
        ["A ball rolls at constant speed. Is there a net force?"],
        ["What happens during puberty?"],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(mcp_server=True)
