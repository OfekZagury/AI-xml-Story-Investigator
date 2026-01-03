import os
from typing import Optional
from openai import OpenAI, OpenAIError


class LLMClient:
    """
    A client wrapper for interacting with the OpenAI API.
    Handles authentication and basic text completion requests.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI client.

        Args:
            api_key (Optional[str]): The OpenAI API key. If not provided,
                                     the client will attempt to use the OPENAI_API_KEY environment variable.
        """
        # The OpenAI client automatically looks for OPENAI_API_KEY in env vars if api_key is None
        self.client = OpenAI(api_key=api_key)
        self.max_input_chars = 3000

    def get_completion(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """
        Sends a prompt to the OpenAI API and retrieves the generated response.

        IMPORTANT: This method enforces a hard limit on the input length.
        Inputs exceeding 3000 characters are truncated to avoid excessive token usage or errors.

        Args:
            prompt (str): The input text to send to the model.
            model (str): The model identifier (default: "gpt-4o-mini").

        Returns:
            str: The content of the model's response, or an empty string if an error occurs.
        """
        # Enforce the 3000-character limit
        if len(prompt) > self.max_input_chars:
            # Truncate the prompt
            prompt = prompt[:self.max_input_chars]

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract and return the content from the first choice
            return response.choices[0].message.content.strip()

        except OpenAIError as e:
            # Handle API errors gracefully (e.g., rate limits, connection issues)
            print(f"Error communicating with OpenAI API: {e}")
            return ""
        except Exception as e:
            # Handle unexpected errors
            print(f"Unexpected error in LLMClient: {e}")
            return ""