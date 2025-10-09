"""
Google Gemini 2.0 Flash integration service.
Handles text generation, embeddings, and streaming responses.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from ..core.config import settings
from .llm_service_interface import LLMServiceInterface, LLMGenerationError, LLMEmbeddingError, LLMConnectionError


class GeminiService(LLMServiceInterface):
    """Service for interacting with Google Gemini 2.0 Flash"""

    def __init__(self):
        # Configure Gemini with API key
        genai.configure(api_key=settings.GEMINI_API_KEY)

        # Initialize models
        self.text_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )

        # For embeddings, we'll use the text embedding model
        self.embedding_model = "models/text-embedding-004"

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[str] = None
    ) -> str:
        """
        Generate a text response using Gemini

        Args:
            prompt: User prompt
            system_prompt: System instruction (prepended to prompt)
            temperature: Response creativity (0.0-1.0)
            max_tokens: Maximum response length
            response_format: Optional format hint ("json" for JSON mode, None for text)

        Returns:
            Generated text response
        """
        try:
            # Combine system prompt and user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

            # Configure generation for this request
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            # Create a new model instance with specific config for this request
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-exp",
                generation_config=generation_config,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )

            # Generate response
            response = await asyncio.to_thread(model.generate_content, full_prompt)

            return response.text

        except Exception as e:
            print(f"Error generating response with Gemini: {e}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"

    async def generate_streaming_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response using Gemini

        Args:
            prompt: User prompt
            system_prompt: System instruction
            temperature: Response creativity
            max_tokens: Maximum response length

        Yields:
            Text chunks as they're generated
        """
        try:
            # Combine prompts
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

            # Configure generation
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-exp",
                generation_config=generation_config,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )

            # Generate streaming response
            response = await asyncio.to_thread(
                model.generate_content,
                full_prompt,
                stream=True
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            yield f"Error: {str(e)}"

    async def generate_embeddings(
        self,
        texts: List[str],
        task_type: str = "retrieval_document"
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed
            task_type: Task type hint ("retrieval_document", "retrieval_query", "classification")

        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []

            for text in texts:
                # Generate embedding for each text
                result = await asyncio.to_thread(
                    genai.embed_content,
                    model=self.embedding_model,
                    content=text,
                    task_type=task_type
                )
                embeddings.append(result['embedding'])

            return embeddings

        except Exception as e:
            raise LLMEmbeddingError(f"Error generating embeddings with Gemini: {e}") from e

    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a search query

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        try:
            result = await asyncio.to_thread(
                genai.embed_content,
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )

            return result['embedding']

        except Exception as e:
            print(f"Error generating query embedding with Gemini: {e}")
            return [0.0] * 768  # Return zero vector as fallback

    async def analyze_content(
        self,
        content: str,
        analysis_prompt: str,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Analyze content using Gemini (for evaluation, classification, etc.)

        Args:
            content: Content to analyze
            analysis_prompt: Instructions for analysis
            temperature: Sampling temperature (default 0.1 for deterministic analysis)

        Returns:
            Analysis results
        """
        try:
            full_prompt = f"{analysis_prompt}\n\nContent to analyze:\n{content}\n\nAnalysis:"

            response = await self.generate_response(
                prompt=full_prompt,
                temperature=temperature
            )

            return {
                "analysis": response,
                "content": content,
                "prompt": analysis_prompt
            }

        except Exception as e:
            return {
                "error": str(e),
                "analysis": "Analysis failed",
                "content": content,
                "prompt": analysis_prompt
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available Gemini models"""
        try:
            models = list(genai.list_models())
            return {
                "available_models": [
                    {
                        "name": model.name,
                        "display_name": model.display_name,
                        "description": model.description
                    }
                    for model in models
                ],
                "current_text_model": "gemini-2.0-flash-exp",
                "current_embedding_model": self.embedding_model
            }
        except Exception as e:
            return {"error": str(e)}

    async def test_connection(self) -> Dict[str, Any]:
        """Test the connection to Gemini API"""
        try:
            # Test text generation
            test_response = await self.generate_response(
                prompt="Say hello and confirm you're working correctly.",
                temperature=0.1
            )

            # Test embeddings
            test_embedding = await self.generate_query_embedding("test query")

            return {
                "status": "success",
                "text_generation": "working" if test_response else "failed",
                "embeddings": "working" if len(test_embedding) > 0 else "failed",
                "test_response": test_response[:100] + "..." if len(test_response) > 100 else test_response,
                "embedding_dimensions": len(test_embedding)
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "text_generation": "failed",
                "embeddings": "failed"
            }

    # Interface helper methods

    def get_model_name(self) -> str:
        """Get the name of the underlying model"""
        return "gemini-2.0-flash-exp"

    def get_embedding_dimensions(self) -> int:
        """Get the dimensionality of embeddings"""
        return 768  # Gemini text-embedding-004 produces 768-dim vectors

    def supports_function_calling(self) -> bool:
        """Check if this LLM service supports function calling"""
        return True  # Gemini 2.0 supports function calling

    def supports_json_mode(self) -> bool:
        """Check if this LLM service supports guaranteed JSON output"""
        return False  # Gemini doesn't have strict JSON mode, but can follow instructions


# Create a global instance
gemini_service = GeminiService()


# Example usage and testing
async def main():
    """Test the Gemini service"""

    print("🧪 Testing Gemini 2.0 Flash Service...")

    # Test connection
    print("\n1. Testing connection...")
    connection_test = await gemini_service.test_connection()
    print(f"Status: {connection_test['status']}")
    print(f"Response: {connection_test.get('test_response', 'N/A')}")

    # Test text generation
    print("\n2. Testing text generation...")
    response = await gemini_service.generate_response(
        prompt="Explain what makes Gemini 2.0 Flash special in one sentence.",
        temperature=0.7
    )
    print(f"Response: {response}")

    # Test embeddings
    print("\n3. Testing embeddings...")
    embeddings = await gemini_service.generate_embeddings([
        "Hello world",
        "This is a test document"
    ])
    print(f"Generated {len(embeddings)} embeddings with {len(embeddings[0]) if embeddings else 0} dimensions")

    # Test streaming
    print("\n4. Testing streaming response...")
    print("Streaming: ", end="")
    async for chunk in gemini_service.generate_streaming_response(
        prompt="Count from 1 to 5 slowly",
        temperature=0.1
    ):
        print(chunk, end="", flush=True)
    print("\n")

    print("✅ Gemini service testing complete!")


if __name__ == "__main__":
    asyncio.run(main())