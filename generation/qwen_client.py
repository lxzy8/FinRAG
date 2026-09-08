"""
Qwen3.5-9B ke saath baat karne ka client — llama.cpp server ke through,
jo OpenAI-compatible API expose karta hai.

NOTE: Ye code assume karta hai ki llama.cpp server already chal raha hai
(kaggle pe alag se start karenge: `llama-server --model qwen3.5-9b-q4.gguf ...`)
Mac pe abhi sirf structure test karenge, actual model load Kaggle pe hoga.
"""

from openai import OpenAI

# llama.cpp server default port
LLAMA_SERVER_URL = "http://localhost:8080/v1"

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=LLAMA_SERVER_URL,
            api_key="not-needed",  # local server hai, key ki zaroorat nahi
        )
    return _client


def generate_answer(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,  # factual finance QA ke liye low rakhna behtar
) -> str:
    """
    Qwen3.5-9B se answer generate karwata hai, given system + user prompt.
    """
    client = get_client()

    response = client.chat.completions.create(
        model="qwen3.5-9b",  # llama.cpp mein jo bhi name diya ho
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return response.choices[0].message.content


def generate_answer_streaming(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
):
    """
    Streaming version — token-by-token yield karta hai (baad mein Gradio UI ke liye).
    """
    client = get_client()

    stream = client.chat.completions.create(
        model="qwen3.5-9b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content