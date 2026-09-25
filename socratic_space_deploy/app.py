import os
import threading

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import spaces  # noqa: E402 — must precede torch!
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "Susu11/socratic_qwen8b-merged"

SYSTEM_PROMPT = (
    "You are a Socratic Science Tutor for a Grade 10 student. Your goal is to guide the student "
    "to discover concepts through reasoning, NEVER by giving the final answer directly. "
    "Ask EXACTLY ONE question per turn. Keep responses to 1-3 sentences. "
    "If the student is stuck, provide a simpler analogy or break the concept into a smaller step. "
    "State your pedagogical goal inside <plan>...</plan> tags."
)

print(f"Loading tokenizer and model from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
EOS_TOKEN_IDS = list(set(i for i in [tokenizer.eos_token_id, im_end_id, endoftext_id, im_start_id] if i is not None))

STOP_STRINGS = [
    "\nuser",
    "\nUser",
    "\nstudent",
    "\nStudent",
    "\nHuman",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
]

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True,
).to("cuda")
model.eval()
print("Model loaded successfully onto ZeroGPU.")


def _chat_duration(message, history=None, show_plan=False, temperature=0.7, top_p=0.8, max_new_tokens=384, *args, **kwargs) -> int:
    """Dynamically reserve realistic GPU seconds for ZeroGPU scheduler."""
    return int(min(120, 20 + float(max_new_tokens) * 0.08))


def _extract_text(content) -> str:
    """Extract plain string from any Gradio 5/6 message payload (str, list, dict)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return " ".join(parts)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return str(content["content"])
        return str(content)
    return str(content)


def _clean_history_text(content) -> str:
    """Strip any UI formatting or plan tags so conversation context remains pure."""
    text = _extract_text(content)
    if "<details" in text and "</details>" in text:
        text = text.split("</details>")[-1].strip()
    if "<plan>" in text and "</plan>" in text:
        text = text.split("</plan>")[-1].strip()
    return text.strip()


def _build_messages(message, history: list) -> list[dict[str, str]]:
    turns = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or []):
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant"):
            content = turn.get("content")
            cleaned = _clean_history_text(content)
            if cleaned:
                turns.append({"role": turn["role"], "content": cleaned})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            user, assistant = turn
            user_txt = _clean_history_text(user)
            asst_txt = _clean_history_text(assistant)
            if user_txt:
                turns.append({"role": "user", "content": user_txt})
            if asst_txt:
                turns.append({"role": "assistant", "content": asst_txt})
    turns.append({"role": "user", "content": _clean_history_text(message)})
    return turns


def _format_stream(text: str, show_plan: bool = False) -> str:
    """Format or hide <plan> ... </plan> tags based on user preference."""
    for stop_s in STOP_STRINGS:
        if stop_s in text:
            text = text.split(stop_s)[0]

    if "<plan>" in text and "</plan>" in text:
        plan_start = text.find("<plan>") + len("<plan>")
        plan_end = text.find("</plan>")
        plan = text[plan_start:plan_end].strip()
        reply = text[plan_end + len("</plan>"):].strip()
        if show_plan:
            quoted_plan = "\n> ".join(plan.splitlines())
            return f"<details open>\n<summary>💡 <b>Tutor Pedagogical Strategy</b></summary>\n\n> {quoted_plan}\n</details>\n\n{reply}"
        else:
            return reply
    elif "<plan>" in text:
        if show_plan:
            plan_partial = text.split("<plan>")[-1].strip()
            quoted_partial = "\n> ".join(plan_partial.splitlines())
            return f"💡 *Formulating Socratic guidance...*\n> {quoted_partial}"
        else:
            return "Thinking..."
    return text


@spaces.GPU(duration=_chat_duration)
def chat(
    message: str,
    history: list,
    show_plan: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_new_tokens: int = 384,
    repetition_penalty: float = 1.15,
):
    """Reply as the Grade 10 Socratic Science Tutor (Qwen2.5-7B fine-tuned).

    Args:
        message: Student's question or response.
        history: Conversation history.
        show_plan: Whether to display internal pedagogical strategy tags.
        temperature: Sampling temperature (0.1 to 1.0).
        top_p: Nucleus sampling probability.
        max_new_tokens: Maximum tokens to generate.
        repetition_penalty: Repetition penalty (e.g. 1.15).

    Yields:
        The tutor's streamed response.
    """
    messages = _build_messages(message, history)
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
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in encoded.items()}
    else:
        inputs = {"input_ids": encoded.to("cuda")}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=int(max_new_tokens),
        do_sample=float(temperature) > 0.05,
        temperature=max(float(temperature), 0.05),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=EOS_TOKEN_IDS,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    raw_accum = ""
    for chunk in streamer:
        raw_accum += chunk

        # Stop early if the model attempts to simulate a user/student response
        stop_found = False
        for stop_s in STOP_STRINGS:
            if stop_s in raw_accum:
                raw_accum = raw_accum.split(stop_s)[0]
                stop_found = True
                break

        yield _format_stream(raw_accum, show_plan=show_plan)
        if stop_found:
            break

    thread.join()


demo = gr.ChatInterface(
    fn=chat,
    title="🎓 Grade 10 Socratic Science Tutor",
    description=(
        "**AI Science Tutor powered by fine-tuned Qwen2.5-7B** "
        "([Susu11/socratic_qwen8b](https://huggingface.co/Susu11/socratic_qwen8b)).\n\n"
        "The tutor follows the **Socratic Method**: instead of giving answers away, "
        "it asks targeted guiding questions to help you reason through concepts step-by-step."
    ),
    additional_inputs=[
        gr.Checkbox(value=False, label="Show Tutor Planning (<plan>)", info="Reveal the internal pedagogical goal formulated by the tutor"),
        gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Top-P"),
        gr.Slider(128, 768, value=384, step=32, label="Max New Tokens"),
        gr.Slider(1.0, 1.3, value=1.15, step=0.05, label="Repetition Penalty"),
    ],
    additional_inputs_accordion=gr.Accordion("⚙️ Tutor Settings & Parameters", open=False),
    examples=[
        ["Why do objects float or sink? Is it because heavy things sink and light things float?"],
        ["If velocity is constant, what is the net force acting on the car?"],
        ["Why are plant leaves green?"],
        ["What is the difference between speed and velocity?"],
        ["Why do we experience different seasons throughout the year?"],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(mcp_server=True)
