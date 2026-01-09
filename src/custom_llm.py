import os
import requests
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


class HuggingFaceAPIWrapper(LLM):
    """Custom wrapper for Hugging Face Router API (OpenAI-compatible) using direct HTTP requests."""

    repo_id: str
    api_token: str
    temperature: float = 0.1

    @property
    def _llm_type(self) -> str:
        return "huggingface_router"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # New Router Endpoint (OpenAI-compatible)
        api_url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        # OpenAI-style payload
        payload = {
            "model": self.repo_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 500,
            "stream": False,
        }

        try:
            response = requests.post(
                api_url, headers=headers, json=payload, timeout=120
            )

            if response.status_code != 200:
                short_error = response.text[:200]
                return f"Error {response.status_code} from Router: {short_error}"

            result = response.json()

            # Parse OpenAI-style response
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]

            return str(result)

        except Exception as e:
            return f"Error calling Hugging Face Router: {e}"
