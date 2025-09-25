import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
import openai
from openai import AsyncOpenAI

from app.core.config import settings
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.gemini_service import gemini_service

class RAGService:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
        self.gemini_service = gemini_service  # Use Gemini as primary LLM

    async def generate_response(
        self,
        query: str,
        agent_id: int,
        conversation_history: List[Dict[str, str]] = None,
        system_prompt: str = None,
        agent_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate a response using RAG (Retrieval-Augmented Generation)"""

        try:
            # Step 1: Retrieve relevant context
            context_results = await self.document_processor.search_similar_content(
                query=query,
                agent_id=agent_id,
                top_k=5
            )

            # Step 2: Prepare context for LLM
            context_text = self._format_context(context_results)

            # Step 3: Build conversation messages
            messages = self._build_messages(
                query=query,
                context=context_text,
                conversation_history=conversation_history or [],
                system_prompt=system_prompt or "You are a helpful assistant."
            )

            # Step 4: Generate response
            response = await self._generate_llm_response(messages, agent_config or {})

            # Step 5: Track sources and metadata
            sources = self._extract_sources(context_results)

            return {
                "response": response["content"],
                "sources": sources,
                "context_used": len(context_results),
                "token_usage": response.get("token_usage", {}),
                "model_used": response.get("model", "unknown")
            }

        except Exception as e:
            print(f"Error generating RAG response: {e}")
            return {
                "response": "I apologize, but I'm having trouble accessing my knowledge base right now. Please try again later.",
                "sources": [],
                "context_used": 0,
                "error": str(e)
            }

    def _format_context(self, context_results: List[Dict[str, Any]]) -> str:
        """Format retrieved context for the LLM"""
        if not context_results:
            return "No relevant context found."

        formatted_context = "Relevant information from knowledge base:\n\n"

        for i, result in enumerate(context_results):
            score = result.get("score", 0)
            text = result.get("text", "")
            source = result.get("metadata", {}).get("source", "Unknown")

            formatted_context += f"[Context {i+1}] (Relevance: {score:.2f}) from {source}:\n"
            formatted_context += f"{text}\n\n"

        return formatted_context

    def _build_messages(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]],
        system_prompt: str
    ) -> List[Dict[str, str]]:
        """Build conversation messages for the LLM"""

        # Enhanced system prompt with context
        enhanced_system_prompt = f"""{system_prompt}

You have access to a knowledge base with relevant information. Use this information to answer questions accurately and helpfully.

When answering:
1. Use the provided context when relevant
2. If the context doesn't contain the answer, say so clearly
3. Be conversational and helpful
4. Cite sources when possible

Context Information:
{context}"""

        messages = [{"role": "system", "content": enhanced_system_prompt}]

        # Add conversation history (limit to last 10 messages)
        for message in conversation_history[-10:]:
            messages.append({
                "role": message["role"],
                "content": message["content"]
            })

        # Add current query
        messages.append({"role": "user", "content": query})

        return messages

    async def _generate_llm_response(
        self,
        messages: List[Dict[str, str]],
        agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate response using Gemini 2.0 Flash as primary LLM"""

        try:
            # Extract config parameters
            temperature = agent_config.get("temperature", 0.7)
            max_tokens = agent_config.get("max_tokens", 1000)

            # Convert messages to prompt format for Gemini
            system_prompt = ""
            user_prompt = ""

            for message in messages:
                if message["role"] == "system":
                    system_prompt = message["content"]
                elif message["role"] == "user":
                    user_prompt = message["content"]
                elif message["role"] == "assistant":
                    # For conversation history, append to user prompt
                    user_prompt += f"\n\nPrevious response: {message['content']}"

            # Generate response using Gemini
            response_content = await self.gemini_service.generate_response(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return {
                "content": response_content,
                "model": "gemini-2.0-flash-exp",
                "token_usage": {
                    "prompt_tokens": len(user_prompt.split()) * 1.3,  # Approximate
                    "completion_tokens": len(response_content.split()) * 1.3,  # Approximate
                    "total_tokens": (len(user_prompt) + len(response_content)) * 1.3  # Approximate
                }
            }

        except Exception as e:
            print(f"Error calling Gemini API: {e}")

            # Fallback to OpenAI if available
            if self.openai_client:
                print("Falling back to OpenAI...")
                try:
                    model = agent_config.get("model", "gpt-4o-mini")
                    temperature = agent_config.get("temperature", 0.7)
                    max_tokens = agent_config.get("max_tokens", 1000)

                    response = await self.openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False
                    )

                    return {
                        "content": response.choices[0].message.content,
                        "model": f"{response.model} (fallback)",
                        "token_usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    }
                except Exception as openai_error:
                    print(f"OpenAI fallback also failed: {openai_error}")

            return {
                "content": f"I encountered an error while generating a response: {str(e)}",
                "model": "error",
                "token_usage": {}
            }

    async def generate_streaming_response(
        self,
        query: str,
        agent_id: int,
        conversation_history: List[Dict[str, str]] = None,
        system_prompt: str = None,
        agent_config: Dict[str, Any] = None
    ):
        """Generate a streaming response using RAG with Gemini"""

        try:
            # Step 1: Retrieve context (same as non-streaming)
            context_results = await self.document_processor.search_similar_content(
                query=query,
                agent_id=agent_id,
                top_k=5
            )

            # Step 2: Prepare context and messages
            context_text = self._format_context(context_results)
            messages = self._build_messages(
                query=query,
                context=context_text,
                conversation_history=conversation_history or [],
                system_prompt=system_prompt or "You are a helpful assistant."
            )

            # Step 3: Send initial metadata
            sources = self._extract_sources(context_results)
            yield {
                "type": "metadata",
                "sources": sources,
                "context_used": len(context_results)
            }

            # Step 4: Convert messages for Gemini
            temperature = agent_config.get("temperature", 0.7) if agent_config else 0.7
            max_tokens = agent_config.get("max_tokens", 1000) if agent_config else 1000

            gemini_system_prompt = ""
            gemini_user_prompt = ""

            for message in messages:
                if message["role"] == "system":
                    gemini_system_prompt = message["content"]
                elif message["role"] == "user":
                    gemini_user_prompt = message["content"]
                elif message["role"] == "assistant":
                    gemini_user_prompt += f"\n\nPrevious response: {message['content']}"

            # Step 5: Generate streaming response with Gemini
            try:
                async for chunk in self.gemini_service.generate_streaming_response(
                    prompt=gemini_user_prompt,
                    system_prompt=gemini_system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                ):
                    yield {
                        "type": "content",
                        "content": chunk
                    }

            except Exception as gemini_error:
                print(f"Gemini streaming failed: {gemini_error}")

                # Fallback to OpenAI streaming if available
                if self.openai_client:
                    print("Falling back to OpenAI streaming...")
                    model = agent_config.get("model", "gpt-4o-mini") if agent_config else "gpt-4o-mini"

                    stream = await self.openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True
                    )

                    async for chunk in stream:
                        if chunk.choices[0].delta.content:
                            yield {
                                "type": "content",
                                "content": chunk.choices[0].delta.content
                            }
                else:
                    yield {"type": "error", "content": f"Streaming failed: {str(gemini_error)}"}
                    return

            # Send completion signal
            yield {"type": "done"}

        except Exception as e:
            yield {"type": "error", "content": str(e)}

    def _extract_sources(self, context_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from context results"""
        sources = []
        seen_sources = set()

        for result in context_results:
            metadata = result.get("metadata", {})
            source = metadata.get("source", "Unknown")

            if source not in seen_sources:
                sources.append({
                    "source": source,
                    "relevance_score": result.get("score", 0),
                    "chunk_index": metadata.get("chunk_index", 0)
                })
                seen_sources.add(source)

        return sources

    async def get_agent_knowledge_stats(self, agent_id: int) -> Dict[str, Any]:
        """Get statistics about an agent's knowledge base"""
        # This would query the database for document counts, vector counts, etc.
        # For now, return mock data
        return {
            "document_count": 5,
            "total_chunks": 150,
            "knowledge_base_size": "2.3 MB",
            "last_updated": "2024-01-01T12:00:00Z"
        }