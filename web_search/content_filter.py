"""
MiniCPM5-1B se web search results ko filter/summarize karna —
noise hatana, sirf query-relevant content Qwen tak pahunchana.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

_model = None
_tokenizer = None

MODEL_ID = "openbmb/MiniCPM5-1B"


def get_filter_model():
    """Lazy load — pehli baar call hone pe hi model load hoga."""
    global _model, _tokenizer
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading MiniCPM5-1B on {device}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype="auto", device_map="auto"
        )
    return _model, _tokenizer


def filter_and_summarize(query: str, search_results: list[dict]) -> str:
    """
    Web search results ko dekh ke sirf relevant portions extract/summarize
    karta hai, taaki Qwen ko clean, condensed context mile.
    """
    if not search_results:
        return "No web results found."

    model, tokenizer = get_filter_model()

    # Saare results ko ek prompt mein combine karo
    results_text = "\n\n".join(
        f"[{i+1}] {r['title']}\n{r['snippet']}\nSource: {r['url']}"
        for i, r in enumerate(search_results)
    )

    prompt = f"""You are given web search results for the question: "{query}"

Search results:
{results_text}

Extract only the information relevant to answering the question. \
Discard irrelevant results. Keep source URLs for anything you keep. \
Be concise."""

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,  # fast filtering ke liye, deep reasoning ki zaroorat nahi
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=400)
    result = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )

    return result