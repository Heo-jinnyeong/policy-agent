"""Ollama chat model adapter for policy analysis."""
import httpx
from langsmith import traceable

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


class OllamaLLM:
    def __init__(self, model: str, usd_per_1k_tokens: float):
        self.name = f"ollama:{model}"
        self.model = model
        self.usd_per_1k = usd_per_1k_tokens

    @traceable(run_type="llm", name="ollama.policy_chat")
    def _chat(self, system: str, user: str, json_mode: bool = False):
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "options": {"temperature": 0}, "stream": False, "think": False}
        if json_mode:
            payload["format"] = "json"
        response = httpx.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        tokens = (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
        return data["message"]["content"], {
            "tokens": tokens, "usd": round(tokens / 1000 * self.usd_per_1k, 6)}


class OllamaFastLLM(OllamaLLM):
    pass


class OllamaQualityLLM(OllamaLLM):
    pass
