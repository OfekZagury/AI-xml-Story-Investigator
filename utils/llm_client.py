import os
from typing import List, Optional

from openai import OpenAI, OpenAIError

# Gemini exposes an OpenAI-compatible endpoint, so one client covers both providers.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"


class LLMClient:
    """
    A client wrapper for chat completions and embeddings.

    Uses Google Gemini when GEMINI_API_KEY is set, otherwise OpenAI (OPENAI_API_KEY).
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key (Optional[str]): An explicit OpenAI API key. When omitted, the provider
                                     is chosen from the environment (Gemini first).
        """
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and not api_key:
            self.provider = "gemini"
            self.client = OpenAI(api_key=gemini_key, base_url=GEMINI_BASE_URL)
            self.chat_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
            self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_GEMINI_EMBEDDING_MODEL)
        else:
            # The OpenAI client reads OPENAI_API_KEY from the environment if api_key is None
            self.provider = "openai"
            self.client = OpenAI(api_key=api_key)
            self.chat_model = os.getenv("OPENAI_MODEL", "gpt-5")
            self.embedding_model = "text-embedding-3-small"
        self.max_input_chars = 3000

    def get_completion(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Sends a prompt to the chat model and returns the response text.

        IMPORTANT: This method enforces a hard limit on the input length.
        Inputs exceeding 3000 characters are truncated to avoid excessive token usage or errors.

        Returns:
            str: The content of the model's response, or an empty string if an error occurs.
        """
        if len(prompt) > self.max_input_chars:
            prompt = prompt[:self.max_input_chars]

        try:
            response = self.client.chat.completions.create(
                model=model or self.chat_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return (response.choices[0].message.content or "").strip()

        except OpenAIError as e:
            # Handle API errors gracefully (e.g., rate limits, connection issues)
            print(f"Error communicating with the {self.provider} API: {e}")
            return ""
        except Exception as e:
            print(f"Unexpected error in LLMClient: {e}")
            return ""

    def get_embeddings(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """Embeds a list of texts (in batches) and returns one vector per text."""
        vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = [t.replace("\n", " ") for t in texts[start:start + batch_size]]
            response = self.client.embeddings.create(input=batch, model=self.embedding_model)
            vectors.extend(item.embedding for item in response.data)
        return vectors
