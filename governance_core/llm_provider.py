"""
Abstract LLM Provider Interface

Supports multiple backends:
- Ollama (default, local, offline-capable)
- Anthropic Claude (optional)
- OpenAI (optional)

Design principles:
- Provider abstraction for swappable backends
- Privacy-first: no PII exposure to external APIs
- Audit trail for all LLM interactions
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All providers must implement the same interface to ensure
    consistent behavior across different backends.
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """
        Generate text from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System/instruction prompt
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Generate structured JSON output.
        
        Args:
            prompt: User prompt
            system_prompt: System/instruction prompt
            schema: Optional JSON schema for validation
            **kwargs: Provider-specific parameters
            
        Returns:
            Parsed JSON response
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured correctly."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        pass


class OllamaProvider(LLMProvider):
    """
    Local LLM provider using Ollama.
    
    Default provider - fully local, offline-capable, no external data exposure.
    
    Installation:
        1. Install Ollama: https://ollama.ai
        2. Pull a model: ollama pull llama3.1:70b
        
    Recommended models:
        - llama3.1:70b (best reasoning)
        - llama3.1:8b (fast, good for CI)
        - qwen2.5:32b (excellent for technical tasks)
    """
    
    DEFAULT_MODEL = "llama3.1:70b"
    FALLBACK_MODEL = "llama3.1:8b"
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        """
        Initialize Ollama provider.
        
        Args:
            model: Model name (e.g., "llama3.1:70b")
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        
        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError(
                "requests library required for Ollama. "
                "Install with: pip install requests"
            )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """Generate text using Ollama."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        try:
            response = self._requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """Generate structured JSON using Ollama."""
        # Enhance prompt to request JSON output
        json_instruction = (
            "\n\nYou MUST respond with valid JSON only. "
            "Do not include any text before or after the JSON object."
        )
        
        if schema:
            json_instruction += f"\n\nJSON Schema:\n{json.dumps(schema, indent=2)}"
        
        full_prompt = prompt + json_instruction
        
        response_text = self.generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            **kwargs
        )
        
        # Extract JSON from response (handle markdown code blocks)
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response_text[:200]}")
            raise ValueError(f"Invalid JSON response: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = self._requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                
                # Check if our model is available
                if self.model in model_names:
                    return True
                
                # Try fallback model
                if self.FALLBACK_MODEL in model_names:
                    logger.warning(
                        f"Model {self.model} not found, using fallback {self.FALLBACK_MODEL}"
                    )
                    self.model = self.FALLBACK_MODEL
                    return True
                
                logger.error(
                    f"Neither {self.model} nor {self.FALLBACK_MODEL} found in Ollama. "
                    f"Available models: {model_names}"
                )
                return False
            return False
        except Exception as e:
            logger.error(f"Ollama availability check failed: {e}")
            return False
    
    @property
    def provider_name(self) -> str:
        return f"Ollama ({self.model})"


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider (optional).
    
    Requires API key and network connectivity.
    Not recommended for production due to external data exposure.
    """
    
    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Model name
        """
        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError:
            raise ImportError(
                "anthropic library required. Install with: pip install anthropic"
            )
        
        import os
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key."
            )
        
        self.model = model
        self.client = self._anthropic.Anthropic(api_key=self.api_key)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """Generate text using Claude."""
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=messages
        )
        
        return response.content[0].text
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """Generate structured JSON using Claude."""
        json_instruction = (
            "\n\nYou MUST respond with valid JSON only. "
            "Do not include any text before or after the JSON object."
        )
        
        if schema:
            json_instruction += f"\n\nJSON Schema:\n{json.dumps(schema, indent=2)}"
        
        full_prompt = prompt + json_instruction
        response_text = self.generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            **kwargs
        )
        
        # Extract JSON
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        return json.loads(response_text)
    
    def is_available(self) -> bool:
        """Check if API key is valid."""
        return bool(self.api_key)
    
    @property
    def provider_name(self) -> str:
        return f"Anthropic ({self.model})"


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider (optional).
    
    Requires API key and network connectivity.
    Not recommended for production due to external data exposure.
    """
    
    DEFAULT_MODEL = "gpt-4o"
    
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name
        """
        try:
            import openai
            self._openai = openai
        except ImportError:
            raise ImportError(
                "openai library required. Install with: pip install openai"
            )
        
        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )
        
        self.model = model
        self.client = self._openai.OpenAI(api_key=self.api_key)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """Generate text using OpenAI."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """Generate structured JSON using OpenAI."""
        # Use response_format for JSON mode
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        json_instruction = (
            "\n\nYou MUST respond with valid JSON only. "
            "Do not include any text before or after the JSON object."
        )
        if schema:
            json_instruction += f"\n\nJSON Schema:\n{json.dumps(schema, indent=2)}"
        
        messages.append({"role": "user", "content": prompt + json_instruction})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def is_available(self) -> bool:
        """Check if API key is valid."""
        return bool(self.api_key)
    
    @property
    def provider_name(self) -> str:
        return f"OpenAI ({self.model})"


def create_provider(
    provider_type: str = "ollama",
    **kwargs
) -> LLMProvider:
    """
    Factory function to create LLM provider instances.
    
    Args:
        provider_type: "ollama" (default), "anthropic", or "openai"
        **kwargs: Provider-specific parameters
        
    Returns:
        LLMProvider instance
        
    Example:
        >>> provider = create_provider("ollama", model="llama3.1:70b")
        >>> response = provider.generate("Explain leakage detection")
    """
    providers = {
        "ollama": OllamaProvider,
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
    }
    
    if provider_type not in providers:
        raise ValueError(
            f"Unknown provider: {provider_type}. "
            f"Available: {list(providers.keys())}"
        )
    
    provider_class = providers[provider_type]
    return provider_class(**kwargs)
