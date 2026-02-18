"""Custom LangChain LLM wrapper for the HuggingFace Router API.

Provides a direct HTTP-based integration with the HuggingFace Router
(OpenAI-compatible endpoint), bypassing the occasionally buggy
``langchain-huggingface`` ``HuggingFaceEndpoint`` class.
"""

import requests
from typing import Any, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from src.config import LLM_MAX_TOKENS, LLM_TIMEOUT_SECONDS


class HuggingFaceAPIWrapper(LLM):
    """Custom wrapper for Hugging Face Router API (OpenAI-compatible).

    Uses direct HTTP ``POST`` requests to the HuggingFace Router endpoint
    to invoke large language models without relying on the LangChain
    HuggingFace integration, which can be unstable.

    Attributes:
        repo_id: HuggingFace model repository identifier
            (e.g. ``"deepseek-ai/DeepSeek-R1"``).
        api_token: Bearer token for authenticating with the
            HuggingFace API.
        temperature: Sampling temperature for generation.
    """

    repo_id: str
    api_token: str
    temperature: float = 0.1

    @property
    def _llm_type(self) -> str:
        """Return a human-readable identifier for this LLM type."""
        return "huggingface_router"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Send a prompt to the HuggingFace Router and return the response.

        Args:
            prompt: The user prompt to send to the model.
            stop: Optional list of stop sequences (currently unused by
                the Router endpoint but accepted for LangChain compatibility).
            run_manager: Optional callback manager for LangChain tracing.
            **kwargs: Additional keyword arguments forwarded to the API.

        Returns:
            The generated text from the model, or a descriptive error
            string if the API call fails.
        """
        api_url: str = "https://router.huggingface.co/v1/chat/completions"
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.repo_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": LLM_MAX_TOKENS,
            "stream": False,
        }

        try:
            response = requests.post(
                api_url, headers=headers, json=payload, timeout=LLM_TIMEOUT_SECONDS
            )

            if response.status_code != 200:
                short_error: str = response.text[:200]
                return f"Error {response.status_code} from Router: {short_error}"

            result: Dict[str, Any] = response.json()

            # Parse OpenAI-style response
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]

            return str(result)

        except requests.exceptions.Timeout:
            return (
                f"Error: HuggingFace Router request timed out after "
                f"{LLM_TIMEOUT_SECONDS}s."
            )
        except requests.exceptions.ConnectionError:
            return "Error: Unable to connect to the HuggingFace Router API."
        except Exception as e:
            return f"Error calling Hugging Face Router: {e}"
