import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
import openai
from openai import AsyncOpenAI

from app.core.config import settings
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.gemini_service import gemini_service
from app.services.personality_service import personality_service
from app.services.context.context_compression import compress_context_snippet
from app.services.reranker_service import reranker_service

class RAGService:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
        self.gemini_service = gemini_service  # Use Gemini as primary LLM
        self.reranker = reranker_service

    async def generate_response(
        self,
        query: str,
        agent_id: int,
        conversation_history: List[Dict[str, str]] = None,
        system_prompt: str = None,
        agent_config: Dict[str, Any] = None,
        db_session = None
    ) -> Dict[str, Any]:
        """Generate a response using RAG (Retrieval-Augmented Generation)"""

        try:
            # Step 1: Retrieve relevant context (increased from 5 to 20 for better recall)
            context_results = await self.document_processor.search_similar_content(
                query=query,
                agent_id=agent_id,
                top_k=20  # Retrieve more candidates for reranking
            )

            # Step 2: Rerank to get best 5 chunks (improved precision)
            context_results = await self.reranker.rerank(
                query,
                context_results or [],
                top_k=5  # Final refined set
            )

            # Step 2: Prepare context for LLM
            context_text = self._format_context(context_results)

            # Step 3: Get agent personality and enhance system prompt
            personality = None
            enhanced_system_prompt = system_prompt or "You are a helpful assistant."

            if db_session:
                personality = await personality_service.get_agent_personality(agent_id, db_session)
                enhanced_system_prompt = personality_service.inject_personality_into_prompt(
                    base_prompt=enhanced_system_prompt,
                    personality=personality,
                    user_query=query,
                    context=context_text
                )

            # Step 4: Build conversation messages
            messages = self._build_messages(
                query=query,
                context=context_text,
                conversation_history=conversation_history or [],
                system_prompt=enhanced_system_prompt
            )

            # Step 5: Generate response
            response = await self._generate_llm_response(messages, agent_config or {})

            # Step 6: Apply personality enhancement to response
            enhanced_response_content = response["content"]
            if personality:
                enhanced_response_content = personality_service.enhance_response_with_personality(
                    response=enhanced_response_content,
                    personality=personality,
                    user_query=query,
                    conversation_history=conversation_history or []
                )

            # Step 7: Track sources and metadata
            sources = self._extract_sources(context_results)

            return {
                "response": enhanced_response_content,
                "sources": sources,
                "context_used": len(context_results),
                "token_usage": response.get("token_usage", {}),
                "model_used": response.get("model", "unknown"),
                "personality_applied": personality is not None
            }

        except Exception as e:
            print(f"Error generating RAG response: {e}")
            return {
                "response": "I apologize, but I'm having trouble accessing my knowledge base right now. Please try again later.",
                "sources": [],
                "context_used": 0,
                "error": str(e)
            }

    async def retrieve_context(
        self,
        query: str,
        agent_id: int,
        top_k: int = 5,
        *,
        return_full: bool = False
    ) -> List[Any]:
        """Retrieve similar knowledge base chunks for a given query.

        Args:
            query: Natural language query to search with.
            agent_id: Agent whose knowledge base should be queried.
            top_k: Maximum number of chunks to return.
            return_full: When True, include score and metadata for each chunk.

        Returns:
            List of chunk texts by default, or rich dictionaries when
            ``return_full`` is True. Falls back gracefully if no results are
            available or an error occurs.
        """

        try:
            raw_results = await self.document_processor.search_similar_content(
                query=query,
                agent_id=agent_id,
                top_k=top_k
            )
        except Exception as exc:
            print(f"Error retrieving context for query '{query}': {exc}")
            return []

        normalized_results: List[Dict[str, Any]] = []

        for result in raw_results or []:
            if isinstance(result, dict):
                text = result.get("text") or ""
                score = result.get("score")
                metadata = result.get("metadata") or {}
            else:
                # Fallback to simple string representation
                text = str(result)
                score = None
                metadata = {}

            chunk_id = metadata.get("chunk_id")

            normalized_results.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": score,
                    "metadata": metadata
                }
            )

        if return_full:
            return await self.reranker.rerank(query, normalized_results, top_k=top_k)

        simple_results: List[str] = []
        reranked = await self.reranker.rerank(query, normalized_results, top_k=top_k)
        for item in reranked:
            if item["text"]:
                simple_results.append(item["text"])
            elif item["chunk_id"]:
                simple_results.append(item["chunk_id"])

        return simple_results

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation: 1 token ≈ 4 chars)"""
        return len(text) // 4

    def _format_context(
        self,
        context_results: List[Dict[str, Any]],
        max_tokens: int = 2000
    ) -> str:
        """Format retrieved context for the LLM with token budget management

        Args:
            context_results: List of search results sorted by relevance
            max_tokens: Maximum tokens to use for context (default: 2000)

        Returns:
            Formatted context string that fits within token budget
        """
        if not context_results:
            return "No relevant context found."

        formatted_context = "Relevant information from knowledge base:\n\n"
        header_tokens = self._estimate_tokens(formatted_context)
        tokens_used = header_tokens
        chunks_included = 0

        # Sort by relevance score (highest first)
        sorted_results = sorted(
            context_results,
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        for i, result in enumerate(sorted_results):
            score = result.get("score", 0)
            text = result.get("text", "")
            metadata = result.get("metadata", {}) or {}
            source = metadata.get("source") or metadata.get("source_url") or "Unknown"
            compressed_text = compress_context_snippet(text, metadata)

            # Format this chunk
            chunk_header = f"[Context {chunks_included + 1}] (Relevance: {score:.2f}) from {source}:\n"
            chunk_content = f"{compressed_text}\n\n"
            chunk_full = chunk_header + chunk_content

            # Check if adding this chunk would exceed budget
            chunk_tokens = self._estimate_tokens(chunk_full)

            if tokens_used + chunk_tokens > max_tokens:
                # Skip remaining chunks if budget exhausted
                break

            formatted_context += chunk_full
            tokens_used += chunk_tokens
            chunks_included += 1

        # Add summary if some chunks were skipped
        if chunks_included < len(sorted_results):
            formatted_context += f"\n[Note: {len(sorted_results) - chunks_included} additional sources available but omitted due to context length]\n"

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

You have access to relevant information from our knowledge base to help answer questions accurately.

Guidelines for using this information:
• Reference the context naturally when it's relevant to the customer's question
• If the available information doesn't fully answer their question, be honest about limitations while staying helpful
• Keep your responses conversational and engaging
• When referencing specific details, you can mention the source naturally (e.g., "According to our product guide...")

IMPORTANT: Format your entire response in Markdown. Use proper Markdown syntax for:
- **Bold text** for emphasis
- *Italic text* for subtle emphasis
- # Headers when appropriate
- - Bullet points for lists
- 1. Numbered lists when showing steps
- `code blocks` for technical terms or code
- > Blockquotes for important notes
- [Links](url) when referencing sources

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
        agent_config: Dict[str, Any] = None,
        db_session = None
    ):
        """Generate a streaming response using RAG with Gemini"""

        try:
            # Step 1: Retrieve context (increased from 5 to 20 for better recall)
            context_results = await self.document_processor.search_similar_content(
                query=query,
                agent_id=agent_id,
                top_k=20  # Retrieve more candidates for reranking
            )

            # Step 1.5: Rerank to get best 5 chunks
            context_results = await self.reranker.rerank(
                query,
                context_results or [],
                top_k=5  # Final refined set
            )

            # Step 2: Prepare context and personality enhancement
            context_text = self._format_context(context_results)

            # Get agent personality and enhance system prompt
            personality = None
            enhanced_system_prompt = system_prompt or "You are a helpful assistant."

            if db_session:
                personality = await personality_service.get_agent_personality(agent_id, db_session)
                enhanced_system_prompt = personality_service.inject_personality_into_prompt(
                    base_prompt=enhanced_system_prompt,
                    personality=personality,
                    user_query=query,
                    context=context_text
                )

            messages = self._build_messages(
                query=query,
                context=context_text,
                conversation_history=conversation_history or [],
                system_prompt=enhanced_system_prompt
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
