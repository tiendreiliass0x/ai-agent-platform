"""
LLM Service Interface - Standardized interface for all LLM providers

This interface ensures consistent LLM integration across all Context Engine
Phase 2 components (Query Understanding, Answer Synthesis, Contradiction Detection, etc.)

Supported Implementations:
    - GeminiService (Google Gemini 2.0 Flash)
    - OpenAIService (GPT-4, GPT-4-Turbo)
    - Future: Claude, Llama, etc.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncGenerator


class LLMServiceInterface(ABC):
    """Standard interface that all LLM services must implement"""

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[str] = None
    ) -> str:
        """
        Generate a text response from the LLM.

        Args:
            prompt: User prompt or query
            system_prompt: Optional system instruction (prepended to prompt)
            temperature: Sampling temperature (0.0-1.0). Lower = more deterministic
            max_tokens: Maximum response length in tokens
            response_format: Optional format hint ("json" for JSON mode, None for text)

        Returns:
            Generated text response as string

        Raises:
            Exception: If generation fails
        """
        pass

    @abstractmethod
    async def generate_streaming_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming text response from the LLM.

        Args:
            prompt: User prompt or query
            system_prompt: Optional system instruction
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum response length in tokens

        Yields:
            Text chunks as they're generated

        Raises:
            Exception: If generation fails
        """
        pass

    @abstractmethod
    async def generate_embeddings(
        self,
        texts: List[str],
        task_type: str = "retrieval_document"
    ) -> List[List[float]]:
        """
        Generate embedding vectors for a list of texts.

        Args:
            texts: List of text strings to embed
            task_type: Task type hint ("retrieval_document", "retrieval_query", "classification")

        Returns:
            List of embedding vectors (one per input text)

        Raises:
            Exception: If embedding generation fails
        """
        pass

    @abstractmethod
    async def generate_query_embedding(
        self,
        query: str
    ) -> List[float]:
        """
        Generate an embedding vector for a search query.

        Args:
            query: Search query text

        Returns:
            Query embedding vector

        Raises:
            Exception: If embedding generation fails
        """
        pass

    @abstractmethod
    async def analyze_content(
        self,
        content: str,
        analysis_prompt: str,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Analyze content using the LLM (for evaluation, classification, extraction).

        Args:
            content: Content to analyze
            analysis_prompt: Instructions for the analysis task
            temperature: Sampling temperature (default 0.1 for deterministic analysis)

        Returns:
            Dictionary containing:
                - analysis: LLM's analysis result
                - content: Original content analyzed
                - prompt: The analysis prompt used

        Raises:
            Exception: If analysis fails
        """
        pass

    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the LLM service.

        Returns:
            Dictionary containing:
                - status: "success" or "error"
                - text_generation: "working" or "failed"
                - embeddings: "working" or "failed"
                - test_response: Sample response (truncated)
                - embedding_dimensions: Dimension count

        Raises:
            Exception: If connection test fails
        """
        pass

    # Helper methods that can have default implementations

    def get_model_name(self) -> str:
        """
        Get the name of the underlying model.

        Returns:
            Model name/identifier
        """
        return "unknown"

    def get_embedding_dimensions(self) -> int:
        """
        Get the dimensionality of embeddings produced by this service.

        Returns:
            Embedding vector dimension count
        """
        return 768  # Default, should be overridden

    def supports_function_calling(self) -> bool:
        """
        Check if this LLM service supports function/tool calling.

        Returns:
            True if function calling is supported
        """
        return False

    def supports_json_mode(self) -> bool:
        """
        Check if this LLM service supports guaranteed JSON output.

        Returns:
            True if JSON mode is supported
        """
        return False


class LLMServiceError(Exception):
    """Base exception for LLM service errors"""
    pass


class LLMConnectionError(LLMServiceError):
    """Raised when connection to LLM service fails"""
    pass


class LLMGenerationError(LLMServiceError):
    """Raised when text generation fails"""
    pass


class LLMEmbeddingError(LLMServiceError):
    """Raised when embedding generation fails"""
    pass
