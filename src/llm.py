"""
LLM Interface module supporting both local Ollama (llama3.2) and Google Gemini API.
"""

import os
import requests
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OllamaLLM:
    """Wrapper class for local Ollama completion calls (using llama3.2)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip('/')
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2")
        self._ollama_client = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            import ollama
            self._ollama_client = ollama.Client(host=self.base_url)
        except Exception:
            self._ollama_client = None

    def is_available(self) -> bool:
        """Checks if the local Ollama server is accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def generate_response(self, prompt: str, temperature: float = 0.1) -> str:
        if not self.is_available():
            return (
                "⚠️ **Ollama Server is not running locally.**\n\n"
                f"Please install Ollama from [https://ollama.com](https://ollama.com) and start it, "
                f"or switch to **Gemini API** in the sidebar."
            )

        if self._ollama_client is not None:
            try:
                response = self._ollama_client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={"temperature": temperature}
                )
                if response and "response" in response:
                    return response["response"].strip()
            except Exception:
                pass

        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            }
            response = requests.post(url, json=payload, timeout=120)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            elif response.status_code == 404:
                return f"⚠️ **Model `{self.model_name}` not found.** Run `ollama pull {self.model_name}` in terminal."
            else:
                return f"Error from Ollama server: {response.text}"
        except Exception as e:
            return f"Error communicating with local Ollama instance: {str(e)}"


class GeminiLLM:
    """Wrapper class for Google Gemini API completion calls."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self._genai_client = None
        self._legacy_genai = None

        if self.api_key and self.api_key != "your_gemini_api_key_here":
            self._init_client()

    def _init_client(self) -> None:
        try:
            from google import genai
            self._genai_client = genai.Client(api_key=self.api_key)
            return
        except Exception:
            pass

        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=self.api_key)
            self._legacy_genai = legacy_genai
            return
        except Exception:
            pass

    def is_available(self) -> bool:
        """Returns True if a valid Gemini API key is set."""
        key = self.api_key or os.getenv("GEMINI_API_KEY")
        return bool(key and key != "your_gemini_api_key_here")

    def generate_response(self, prompt: str, temperature: float = 0.1) -> str:
        if not self.is_available():
            return (
                "⚠️ **Gemini API Key missing or invalid.**\n\n"
                "Please enter your `GEMINI_API_KEY` in the sidebar or `.env` file."
            )

        if self._genai_client is None and self._legacy_genai is None:
            self.api_key = os.getenv("GEMINI_API_KEY")
            self._init_client()

        if self._genai_client is not None:
            try:
                for m in [self.model_name, "gemini-2.5-flash", "gemini-1.5-flash"]:
                    try:
                        res = self._genai_client.models.generate_content(model=m, contents=prompt)
                        if res and res.text:
                            return res.text.strip()
                    except Exception:
                        continue
            except Exception as e:
                return f"Gemini API Error: {str(e)}"

        if self._legacy_genai is not None:
            try:
                for m in [self.model_name, "gemini-1.5-flash", "gemini-pro"]:
                    try:
                        model = self._legacy_genai.GenerativeModel(m)
                        res = model.generate_content(prompt, generation_config={"temperature": temperature})
                        if res and res.text:
                            return res.text.strip()
                    except Exception:
                        continue
            except Exception as e:
                return f"Gemini API Error: {str(e)}"

        return "Gemini API client could not be initialized."
