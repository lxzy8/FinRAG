"""LLM helper utilities: response-text extraction, retry-with-backoff, call stats."""

import time


def extract_text(response) -> str:
    """
    Gemma kabhi plain string deta hai, kabhi list-of-dicts (thinking + text blocks).
    Sirf 'text' type chahiye, 'thinking' wala internal reasoning nahi.
    """
    if isinstance(response.content, str):
        return response.content
    elif isinstance(response.content, list):
        return "".join(
            part.get("text", "") for part in response.content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(response.content)


class LLMStats:
    """Tracks LLM call count/latency — used for cost & latency reporting (eval/run_eval.py)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_calls = 0
        self.total_time = 0.0
        self.call_log = []

    def record(self, elapsed: float):
        self.total_calls += 1
        self.total_time += elapsed
        self.call_log.append(elapsed)

    def as_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_time": self.total_time,
            "call_log": list(self.call_log),
        }


def invoke_with_retry(llm, prompt, stats: LLMStats = None, max_retries=3, min_gap=2):
    """
    429 (rate limit) errors ke liye exponential backoff.
    min_gap: har call se pehle chhota pacing gap, RPM limit se bachne ke liye.
    stats: optional LLMStats instance — diya ho to har successful call log hoti hai.
    """
    time.sleep(min_gap)
    for attempt in range(max_retries):
        start = time.time()
        try:
            response = llm.invoke(prompt)
            if stats is not None:
                stats.record(time.time() - start)
            return response
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_retries - 1:
                wait_time = 2 ** attempt * 10  # 10s, 20s, 40s
                print(f"Rate limited, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    return None
