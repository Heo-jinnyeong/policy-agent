"""OpenAI chat model adapter for policy analysis."""
import os

import httpx
from langsmith import traceable


class OpenAILLM:
    def __init__(self, model: str, usd_per_1k_tokens: float):
        self.name = f"openai:{model}"
        self.model = model
        self.usd_per_1k = usd_per_1k_tokens

    @traceable(run_type="llm", name="openai.policy_chat")
    def _chat(self, system: str, user: str, json_mode: bool = False):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM_BACKEND=openai")
        payload = {"model": self.model, "temperature": 0,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = httpx.post("https://api.openai.com/v1/chat/completions", json=payload,
                              headers={"Authorization": f"Bearer {key}"}, timeout=120)
        response.raise_for_status()
        data = response.json()
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return data["choices"][0]["message"]["content"], {
            "tokens": tokens, "usd": round(tokens / 1000 * self.usd_per_1k, 6)}


class OpenAIFastLLM(OpenAILLM):
    pass


class OpenAIQualityLLM(OpenAILLM):
    pass
