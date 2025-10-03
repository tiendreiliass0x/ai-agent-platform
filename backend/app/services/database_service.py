"""
Database service for real data operations.
Replaces mock data with actual database persistence.
"""

from typing import List, Dict, Any, Optional
from collections import Counter
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy import and_, or_, desc, func, delete
import hashlib
import uuid
import secrets
from datetime import datetime, timedelta

from ..core.database import get_db, get_db_session
from ..core.auth import get_password_hash
from ..models.user import User
from ..models.agent import Agent
from ..models.document import Document
from ..models.conversation import Conversation
from ..models.message import Message
from ..models.organization import Organization
from ..models.user_organization import UserOrganization, OrganizationRole


class DatabaseService:
    """Service for database operations with real persistence"""

    def __init__(self):
        pass

    async def get_session(self) -> AsyncSession:
        """Get database session"""
        from ..core.database import get_db_session
        return await get_db_session()

    async def _ensure_agent_public_id(self, agent: Optional[Agent], db: AsyncSession) -> Optional[Agent]:
        """Guarantee agent has a non-guessable public identifier."""
        if agent and not getattr(agent, "public_id", None):
            agent.public_id = str(uuid.uuid4())
            db.add(agent)
            await db.commit()
            await db.refresh(agent)
        return agent

    # User Operations
    async def create_user(
        self,
        email: str,
        password_hash: str,
        name: str,
        plan: str = "free"
    ) -> User:
        """Create a new user.

        Automatically hashes the password if raw text is provided to prevent
        accidental storage of plaintext credentials.
        """

        async with await self.get_session() as db:
            normalized_hash = password_hash or ""
            if not normalized_hash.startswith("$2"):
                normalized_hash = get_password_hash(normalized_hash)

            user = User(
                email=email,
                password_hash=normalized_hash,
                name=name,
                plan=plan,
                is_active=True
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
            return user

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(User).where(User.email == email)
            )
            return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()

    # Agent Operations
    async def create_agent(
        self,
        db: AsyncSession,
        user_id: int,
        organization_id: int,
        name: str,
        description: str,
        system_prompt: str,
        config: Dict[str, Any] = None,
        widget_config: Dict[str, Any] = None,
        idempotency_key: str = None
    ) -> Agent:
        """Create a new agent within a database session

        Args:
            db: Database session (required). Caller controls transaction.
            user_id: ID of user creating the agent
            organization_id: ID of organization owning the agent
            idempotency_key: Optional key for idempotent creation

        Note: Agent is added to session but NOT committed. Caller must commit.
        """
        # Generate unique API key
        api_key = f"agent_{secrets.token_urlsafe(32)}"

        agent = Agent(
            user_id=user_id,
            organization_id=organization_id,
            name=name,
            description=description,
            system_prompt=system_prompt,
            public_id=str(uuid.uuid4()),
            api_key=api_key,
            config=config or {},
            widget_config=widget_config or {},
            idempotency_key=idempotency_key,
            is_active=True,
            total_conversations=0,
            total_messages=0
        )

        db.add(agent)
        await db.flush()  # Get ID without committing
        await db.refresh(agent)
        return agent

    async def get_agent_by_id(self, agent_id: int, db: AsyncSession = None) -> Optional[Agent]:
        """Get agent by ID

        Args:
            agent_id: Agent ID
            db: Optional database session. If not provided, creates a new one.
        """
        if db:
            result = await db.execute(
                select(Agent)
                .options(selectinload(Agent.documents))
                .where(Agent.id == agent_id)
            )
            agent = result.scalar_one_or_none()
            return agent  # Skip ensure_public_id in provided session
        else:
            async with await self.get_session() as db:
                result = await db.execute(
                    select(Agent)
                    .options(selectinload(Agent.documents))
                    .where(Agent.id == agent_id)
                )
                agent = result.scalar_one_or_none()
                return await self._ensure_agent_public_id(agent, db)
            

    async def get_agent_by_api_key(self, api_key: str) -> Optional[Agent]:
        """Get agent by API key"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Agent).where(Agent.api_key == api_key)
            )
            agent = result.scalar_one_or_none()
            return await self._ensure_agent_public_id(agent, db)

    async def get_agent_by_public_id(self, public_id: str) -> Optional[Agent]:
        """Get agent using its public UUID."""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Agent)
                .options(selectinload(Agent.documents))
                .options(selectinload(Agent.organization))
                .where(Agent.public_id == public_id)
            )
            agent = result.scalar_one_or_none()
            return await self._ensure_agent_public_id(agent, db)


    async def get_agent_by_idempotency_key(
        self,
        user_id: int,
        organization_id: int,
        idempotency_key: str,
        db: AsyncSession = None
    ) -> Optional[Agent]:
        """Get agent by idempotency key (scoped to user and organization)"""
        async def _query(session: AsyncSession):
            result = await session.execute(
                select(Agent)
                .where(and_(
                    Agent.user_id == user_id,
                    Agent.organization_id == organization_id,
                    Agent.idempotency_key == idempotency_key
                ))
            )
            return result.scalar_one_or_none()

        if db:
            return await _query(db)
        else:
            async with await self.get_session() as db:
                return await _query(db)

    async def count_organization_agents_in_session(
        self,
        organization_id: int,
        db_session: AsyncSession
    ) -> int:
        """Count agents for organization within a session (for atomic checks)"""
        result = await db_session.execute(
            select(func.count(Agent.id))
            .where(and_(
                Agent.organization_id == organization_id,
                Agent.is_active == True
            ))
        )
        return result.scalar() or 0

    async def get_user_agents(self, user_id: int) -> List[Agent]:
        """Get all agents for a user"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Agent)
                .where(Agent.user_id == user_id)
                .order_by(desc(Agent.created_at))
            )
            agents = result.scalars().all()
            ensured_agents = []
            for agent in agents:
                ensured_agents.append(await self._ensure_agent_public_id(agent, db))
            return ensured_agents

    async def get_organization_agents(
        self,
        organization_id: int,
        db: AsyncSession = None,
        limit: int = None,
        offset: int = 0
    ) -> List[Agent]:
        """Get all agents for an organization with pagination support

        Args:
            organization_id: Organization ID
            db: Optional database session
            limit: Maximum number of agents to return
            offset: Number of agents to skip
        """
        async def _query(session: AsyncSession):
            query = (
                select(Agent)
                .options(
                    selectinload(Agent.persona),
                    selectinload(Agent.knowledge_pack)
                )
                .where(and_(Agent.organization_id == organization_id, Agent.is_active == True))
                .order_by(desc(Agent.created_at))
            )
            if limit:
                query = query.limit(limit).offset(offset)

            result = await session.execute(query)
            return result.scalars().all()

        if db:
            return await _query(db)
        else:
            async with await self.get_session() as db:
                agents = await _query(db)
                ensured_agents = []
                for agent in agents:
                    ensured_agents.append(await self._ensure_agent_public_id(agent, db))
                return ensured_agents

    async def update_agent(
        self,
        agent_id: int,
        **kwargs
    ) -> Optional[Agent]:
        """Update agent"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Agent).where(Agent.id == agent_id)
            )
            agent = result.scalar_one_or_none()

            if agent:
                for key, value in kwargs.items():
                    if hasattr(agent, key):
                        setattr(agent, key, value)

                agent.updated_at = datetime.utcnow()
                await db.commit()
                await db.refresh(agent)

            return agent

    async def delete_agent(self, agent_id: int) -> bool:
        """
        Delete an agent and all associated data.
        This includes conversations, messages, documents, and vector embeddings.
        """
        async with await self.get_session() as db:
            try:
                # Get the agent first to verify it exists
                result = await db.execute(
                    select(Agent).where(Agent.id == agent_id)
                )
                agent = result.scalar_one_or_none()

                if not agent:
                    return False

                # Get all documents for this agent to clean up vector embeddings
                documents = await self.get_agent_documents(agent_id)

                # Clean up vector embeddings first
                from .vector_store import VectorStoreService
                vector_store = VectorStoreService()

                for document in documents:
                    if document.vector_ids:
                        try:
                            await vector_store.delete_vectors(document.vector_ids)
                        except Exception as e:
                            print(f"Warning: Failed to delete vectors for document {document.id}: {e}")

                # Delete by agent_id filter (more efficient than individual deletions)
                try:
                    await vector_store.delete_agent_vectors(agent_id)
                except Exception as e:
                    print(f"Warning: Failed to delete agent vectors: {e}")

                # Delete all related data in proper order (respecting foreign key constraints)

                # 1. Delete messages (they reference conversations)
                # All models are already imported at top of file
                await db.execute(
                    delete(Message).where(
                        Message.conversation_id.in_(
                            select(Conversation.id).where(Conversation.agent_id == agent_id)
                        )
                    )
                )

                # 2. Delete conversations
                await db.execute(
                    delete(Conversation).where(Conversation.agent_id == agent_id)
                )

                # 3. Delete documents
                await db.execute(
                    delete(Document).where(Document.agent_id == agent_id)
                )

                # 4. Delete memory entries (if they exist)
                try:
                    from ..models.memory import CustomerMemory
                    await db.execute(
                        delete(CustomerMemory).where(CustomerMemory.agent_id == agent_id)
                    )
                except ImportError:
                    # Memory model might not exist yet
                    pass

                # 5. Delete escalations (if they exist)
                try:
                    from ..models.escalation import Escalation
                    await db.execute(
                        delete(Escalation).where(Escalation.agent_id == agent_id)
                    )
                except ImportError:
                    # Escalation model might not exist yet
                    pass

                # 6. Finally delete the agent itself
                await db.execute(
                    delete(Agent).where(Agent.id == agent_id)
                )

                await db.commit()
                return True

            except Exception as e:
                await db.rollback()
                print(f"Error deleting agent {agent_id}: {e}")
                return False

    # Document Operations
    async def create_document(
        self,
        agent_id: int,
        filename: str,
        content: str,
        content_type: str,
        doc_metadata: Dict[str, Any] = None,
        *,
        status: str = "pending",
        chunk_count: int = 0,
        vector_ids: Optional[List[str]] = None,
        error_message: Optional[str] = None
    ) -> Document:
        """Create a new document"""
        async with await self.get_session() as db:
            # Generate content hash for deduplication
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            document = Document(
                agent_id=agent_id,
                filename=filename,
                content=content,
                content_type=content_type,
                content_hash=content_hash,
                doc_metadata=doc_metadata or {},
                size=len(content.encode()),
                status=status,
                chunk_count=chunk_count,
                vector_ids=vector_ids or [],
                error_message=error_message
            )
            db.add(document)
            await db.commit()
            await db.refresh(document)
            return document

    async def get_agent_documents(self, agent_id: int) -> List[Document]:
        """Get all documents for an agent"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Document)
                .where(Document.agent_id == agent_id)
                .order_by(desc(Document.created_at))
            )
            return result.scalars().all()

    async def get_documents_by_ids(self, document_ids: List[int]) -> List[Document]:
        """Get documents by a list of IDs"""
        if not document_ids:
            return []
        async with await self.get_session() as db:
            result = await db.execute(
                select(Document)
                .where(Document.id.in_(document_ids))
            )
            return result.scalars().all()

    async def get_document_by_id(self, document_id: int) -> Optional[Document]:
        """Get document by ID"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Document).where(Document.id == document_id)
            )
            return result.scalar_one_or_none()

    async def delete_document(self, document_id: int) -> bool:
        """Delete a document"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Document).where(Document.id == document_id)
            )
            document = result.scalar_one_or_none()

            if document:
                await db.delete(document)
                await db.commit()
                return True
            return False

    async def update_document_processing(
        self,
        document_id: int,
        *,
        status: Optional[str] = None,
        chunk_count: Optional[int] = None,
        vector_ids: Optional[List[str]] = None,
        error_message: Optional[str] = None,
        content: Optional[str] = None,
        doc_metadata_updates: Optional[Dict[str, Any]] = None
    ) -> Optional[Document]:
        """Update document fields after processing."""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Document).where(Document.id == document_id)
            )
            document = result.scalar_one_or_none()

            if not document:
                return None

            if status is not None:
                document.status = status
            if chunk_count is not None:
                document.chunk_count = chunk_count
            if vector_ids is not None:
                document.vector_ids = vector_ids
            if error_message is not None:
                document.error_message = error_message
            if content is not None:
                document.content = content
                document.size = len(content.encode())
                document.content_hash = hashlib.sha256(content.encode()).hexdigest()
            if doc_metadata_updates:
                current_metadata = document.doc_metadata or {}
                current_metadata.update(doc_metadata_updates)
                document.doc_metadata = current_metadata

            await db.commit()
            await db.refresh(document)
            return document

    # Conversation Operations
    async def create_conversation(
        self,
        agent_id: int,
        session_id: str,
        metadata: Dict[str, Any] = None
    ) -> Conversation:
        """Create a new conversation"""
        async with await self.get_session() as db:
            conversation = Conversation(
                agent_id=agent_id,
                session_id=session_id,
                conv_metadata=metadata or {}
            )
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)
            return conversation

    async def get_conversation_by_session(
        self,
        agent_id: int,
        session_id: str
    ) -> Optional[Conversation]:
        """Get conversation by session ID"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Conversation)
                .where(
                    and_(
                        Conversation.agent_id == agent_id,
                        Conversation.session_id == session_id
                    )
                )
            )
            return result.scalar_one_or_none()

    async def get_agent_conversations(
        self,
        agent_id: int,
        limit: int = 20,
        offset: int = 0
    ) -> List[Conversation]:
        """Get paginated recent conversations for an agent"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Conversation)
                .options(selectinload(Conversation.customer_profile))
                .where(Conversation.agent_id == agent_id)
                .order_by(desc(Conversation.updated_at))
                .offset(offset)
                .limit(limit)
            )
            return result.scalars().all()

    # Message Operations
    async def create_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> Message:
        """Create a new message"""
        async with await self.get_session() as db:
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                msg_metadata=metadata or {}
            )
            db.add(message)
            await db.commit()
            await db.refresh(message)

            # Update conversation timestamp
            conv_result = await db.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            conversation = conv_result.scalar_one_or_none()
            if conversation:
                conversation.updated_at = datetime.utcnow()
                await db.commit()

            return message

    async def get_conversation_messages(
        self,
        conversation_id: int,
        limit: int = 100,
        ascending: bool = True
    ) -> List[Message]:
        """Get messages for a conversation"""
        async with await self.get_session() as db:
            order_clause = Message.created_at.asc() if ascending else Message.created_at.desc()
            result = await db.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(order_clause)
                .limit(limit)
            )
            messages = result.scalars().all()
            return messages if ascending else list(reversed(messages))

    async def count_conversation_messages(self, conversation_id: int) -> int:
        """Count messages in a conversation"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
            )
            return result.scalar_one_or_none() or 0

    async def get_conversation_by_id(self, conversation_id: int) -> Optional[Conversation]:
        """Get a conversation with related agent, customer profile, and messages"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Conversation)
                .options(selectinload(Conversation.messages))
                .options(selectinload(Conversation.customer_profile))
                .options(selectinload(Conversation.agent))
                .where(Conversation.id == conversation_id)
            )
            return result.scalar_one_or_none()

    # Analytics Operations
    async def get_agent_stats(self, agent_id: int, time_range: str = "30d") -> Dict[str, Any]:
        """Get comprehensive stats for an agent, including overview and timeseries.

        Returns a structure compatible with the frontend dashboard while keeping
        backward-compatible top-level keys used by internal tests/scripts.
        """
        async with await self.get_session() as db:
            # Time window
            now = datetime.utcnow()
            days = 30
            if isinstance(time_range, str):
                tr = time_range.lower().strip()
                if tr.endswith("d") and tr[:-1].isdigit():
                    days = int(tr[:-1])
                elif tr in ("week", "7"):  # safeguards
                    days = 7
                elif tr in ("quarter", "90"):  # safeguards
                    days = 90
            start_time = now - timedelta(days=days)

            # Base queries
            conversations_q = select(Conversation).where(Conversation.agent_id == agent_id)
            messages_q = (
                select(Message)
                .join(Conversation, Conversation.id == Message.conversation_id)
                .where(Conversation.agent_id == agent_id)
            )
            documents_q = select(Document).where(Document.agent_id == agent_id)

            conversations_result = await db.execute(conversations_q)
            conversations = conversations_result.scalars().all()

            messages_result = await db.execute(messages_q)
            messages = messages_result.scalars().all()

            documents_result = await db.execute(documents_q)
            total_documents = len(documents_result.scalars().all())

            # Unique users (prefer visitor_id from conv_metadata; fallback to session_id prefix)
            unique_visitors = set()
            for conv in conversations:
                visitor_id = None
                try:
                    data = getattr(conv, "conv_metadata", {}) or {}
                    visitor_id = (data or {}).get("visitor_id")
                except Exception:
                    visitor_id = None
                if not visitor_id and conv.session_id:
                    # session_id format: "{visitor_id}_{agent_id}"
                    visitor_id = conv.session_id.rsplit("_", 1)[0]
                if visitor_id:
                    unique_visitors.add(visitor_id)

            # Filter conversations and messages within the time range
            convs_in_range = [c for c in conversations if c.created_at and c.created_at >= start_time]

            unique_visitors_range = set()
            for conv in convs_in_range:
                meta = getattr(conv, "conv_metadata", {}) or {}
                visitor_id = meta.get("visitor_id")
                if not visitor_id and conv.session_id:
                    visitor_id = conv.session_id.rsplit("_", 1)[0]
                if visitor_id:
                    unique_visitors_range.add(visitor_id)

            unique_users = len(unique_visitors_range)

            # Average response time (ms) computed from user->assistant pairs
            # Fallback to 0 if not computable
            response_times: List[int] = []
            conv_ids_in_range = {c.id for c in convs_in_range}
            messages_in_range = [
                m for m in messages
                if m.created_at and m.created_at >= start_time and m.conversation_id in conv_ids_in_range
            ]

            # Group messages by conversation and sort
            msgs_by_conv: Dict[int, List[Message]] = {}
            for m in messages_in_range:
                msgs_by_conv.setdefault(m.conversation_id, []).append(m)
            for conv_id, conv_msgs in msgs_by_conv.items():
                conv_msgs.sort(key=lambda x: x.created_at or now)
                last_user_ts = None
                for m in conv_msgs:
                    if m.role == "user":
                        last_user_ts = m.created_at
                    elif m.role == "assistant" and last_user_ts and m.created_at:
                        delta = (m.created_at - last_user_ts).total_seconds()
                        if delta >= 0:
                            response_times.append(int(delta))
                        last_user_ts = None
            avg_response_time = (sum(response_times) / len(response_times)) if response_times else 0

            # Conversations timeseries per day within range
            day_counts: Dict[str, int] = {}
            for i in range(days):
                day = (start_time + timedelta(days=i)).date().isoformat()
                day_counts[day] = 0
            for c in convs_in_range:
                day = c.created_at.date().isoformat()
                if day in day_counts:
                    day_counts[day] += 1
            timeseries = [
                {"date": day, "count": count} for day, count in sorted(day_counts.items())
            ]

            # Sentiment and source breakout
            sentiment_counter = Counter()
            source_counter = Counter()
            total_tokens = 0
            web_search_sessions = 0

            for conv in convs_in_range:
                meta = (conv.conv_metadata or {}) if hasattr(conv, "conv_metadata") else {}
                last_sentiment = meta.get("last_sentiment")
                label = None
                if isinstance(last_sentiment, dict):
                    label = last_sentiment.get("label")
                elif isinstance(last_sentiment, str):
                    label = last_sentiment
                if label:
                    normalized = label.lower()
                    if normalized in ("positive", "neutral", "negative"):
                        sentiment_counter[normalized] += 1

                sources = meta.get("sources") or []
                for source in sources:
                    if isinstance(source, dict):
                        key = source.get("source_url") or source.get("source") or source.get("title")
                    else:
                        key = str(source)
                    if key:
                        source_counter[key] += 1

                total_tokens += meta.get("total_tokens", 0)
                if meta.get("web_search_used"):
                    web_search_sessions += 1

            # Naive period-over-period deltas (set to 0 by default)
            overview = {
                "totalConversations": len(convs_in_range),
                "totalMessages": len(messages_in_range),
                "uniqueUsers": unique_users,
                "avgResponseTime": int(avg_response_time),
                "conversationsChange": 0,
                "messagesChange": 0,
                "usersChange": 0,
                "responseTimeChange": 0,
                "totalTokens": total_tokens,
                "webSearchSessions": web_search_sessions,
            }

            sentiment_breakdown = {
                "positive": sentiment_counter.get("positive", 0),
                "neutral": sentiment_counter.get("neutral", 0),
                "negative": sentiment_counter.get("negative", 0),
            }

            top_sources = [
                {"source": key, "count": count}
                for key, count in source_counter.most_common(5)
            ]

            # Return both dashboard shape and legacy fields
            return {
                "overview": overview,
                "conversations": timeseries,
                "sentimentBreakdown": sentiment_breakdown,
                "topSources": top_sources,
                "totalTokens": total_tokens,
                "webSearchSessions": web_search_sessions,
                # Legacy fields
                "total_conversations": len(convs_in_range),
                "total_messages": len(messages_in_range),
                "total_documents": total_documents,
                "recent_conversations": len(convs_in_range),
                "agent_id": agent_id,
            }

    async def get_agent_insights(self, agent_id: int, time_range: str = "30d") -> Dict[str, Any]:
        """Compute agent insights for the dashboard (top questions and satisfaction mix)."""
        async with await self.get_session() as db:
            now = datetime.utcnow()
            days = 30
            tr = (time_range or "30d").lower().strip()
            if tr.endswith("d") and tr[:-1].isdigit():
                days = int(tr[:-1])
            start_time = now - timedelta(days=days)

            # Get recent user messages
            result = await db.execute(
                select(Message)
                .join(Conversation, Conversation.id == Message.conversation_id)
                .where(
                    and_(
                        Conversation.agent_id == agent_id,
                        Message.role == "user",
                        Message.created_at >= start_time,
                    )
                )
            )
            user_messages = result.scalars().all()

            # Top questions by exact content frequency (simple heuristic)
            counts = Counter()
            for m in user_messages:
                text = (m.content or "").strip()
                if not text:
                    continue
                # Normalize whitespace and trim to avoid super long bars
                key = " ".join(text.split())[:120]
                counts[key] += 1
            total = sum(counts.values()) or 1
            top = counts.most_common(5)
            top_questions = [
                {"question": q, "count": c, "percentage": round(c / total * 100)} for q, c in top
            ]

            # Rough satisfaction estimation via simple keyword heuristics
            pos_words = ("thank you", "thanks", "great", "awesome", "love", "helpful")
            neg_words = ("ridiculous", "angry", "cancel", "frustrated", "terrible", "bad")
            pos = neg = neu = 0
            for m in user_messages:
                lc = (m.content or "").lower()
                if any(w in lc for w in pos_words):
                    pos += 1
                elif any(w in lc for w in neg_words):
                    neg += 1
                else:
                    neu += 1
            total_msgs = max(pos + neg + neu, 1)
            satisfaction = {
                "positive": int(round(pos / total_msgs * 100)),
                "neutral": int(round(neu / total_msgs * 100)),
                "negative": int(round(neg / total_msgs * 100)),
            }

            return {"topQuestions": top_questions, "satisfaction": satisfaction}

    # Utility Operations
    async def search_agents(
        self,
        user_id: int,
        query: str,
        limit: int = 20
    ) -> List[Agent]:
        """Search agents by name or description"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Agent)
                .where(
                    and_(
                        Agent.user_id == user_id,
                        or_(
                            Agent.name.ilike(f"%{query}%"),
                            Agent.description.ilike(f"%{query}%")
                        )
                    )
                )
                .order_by(desc(Agent.updated_at))
                .limit(limit)
            )
            return result.scalars().all()

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics"""
        async with await self.get_session() as db:
            # Count total users
            users_result = await db.execute(select(User))
            total_users = len(users_result.scalars().all())

            # Count total agents
            agents_result = await db.execute(select(Agent))
            total_agents = len(agents_result.scalars().all())

            # Count active agents
            active_agents_result = await db.execute(
                select(Agent).where(Agent.is_active == True)
            )
            active_agents = len(active_agents_result.scalars().all())

            # Count total documents
            documents_result = await db.execute(select(Document))
            total_documents = len(documents_result.scalars().all())

            # Count total conversations
            conversations_result = await db.execute(select(Conversation))
            total_conversations = len(conversations_result.scalars().all())

            return {
                "total_users": total_users,
                "total_agents": total_agents,
                "active_agents": active_agents,
                "total_documents": total_documents,
                "total_conversations": total_conversations
            }

    # Organization Operations
    async def create_organization(
        self,
        name: str,
        slug: str,
        description: Optional[str] = None,
        website: Optional[str] = None,
        plan: str = "free",
        max_agents: int = 3,
        max_users: int = 1,
        max_documents_per_agent: int = 10
    ) -> Organization:
        """Create a new organization"""
        async with await self.get_session() as db:
            organization = Organization(
                name=name,
                slug=slug,
                description=description,
                website=website,
                plan=plan,
                max_agents=max_agents,
                max_users=max_users,
                max_documents_per_agent=max_documents_per_agent,
                is_active=True,
                subscription_status="active"
            )
            db.add(organization)
            await db.commit()
            await db.refresh(organization)
            return organization

    async def get_organization_by_id(self, organization_id: int) -> Optional[Organization]:
        """Get organization by ID"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Organization).where(Organization.id == organization_id)
            )
            return result.scalar_one_or_none()

    async def get_organization_by_slug(self, slug: str) -> Optional[Organization]:
        """Get organization by slug"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Organization).where(Organization.slug == slug)
            )
            return result.scalar_one_or_none()

    async def update_organization(self, organization_id: int, **kwargs) -> Optional[Organization]:
        """Update organization"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Organization).where(Organization.id == organization_id)
            )
            organization = result.scalar_one_or_none()
            if not organization:
                return None

            for key, value in kwargs.items():
                if hasattr(organization, key):
                    setattr(organization, key, value)

            await db.commit()
            await db.refresh(organization)
            return organization

    async def delete_organization(self, organization_id: int) -> bool:
        """Delete organization"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Organization).where(Organization.id == organization_id)
            )
            organization = result.scalar_one_or_none()
            if not organization:
                return False

            await db.delete(organization)
            await db.commit()
            return True

    # UserOrganization Operations
    async def add_user_to_organization(
        self,
        user_id: int,
        organization_id: int,
        role: str = OrganizationRole.VIEWER.value,
        invited_by_user_id: Optional[int] = None
    ) -> UserOrganization:
        """Add user to organization"""
        async with await self.get_session() as db:
            user_org = UserOrganization(
                user_id=user_id,
                organization_id=organization_id,
                role=role,
                invited_by_user_id=invited_by_user_id,
                is_active=True
            )
            db.add(user_org)
            await db.commit()
            await db.refresh(user_org)
            return user_org

    async def get_user_organization(self, user_id: int, organization_id: int) -> Optional[UserOrganization]:
        """Get user organization relationship"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(UserOrganization).where(
                    and_(
                        UserOrganization.user_id == user_id,
                        UserOrganization.organization_id == organization_id
                    )
                )
            )
            return result.scalar_one_or_none()

    async def get_user_organizations(self, user_id: int) -> List[UserOrganization]:
        """Get all organizations for a user"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(UserOrganization)
                .options(selectinload(UserOrganization.organization))
                .where(
                    and_(
                        UserOrganization.user_id == user_id,
                        UserOrganization.is_active == True
                    )
                )
                .order_by(UserOrganization.created_at)
            )
            return result.scalars().all()

    async def get_organization_members(self, organization_id: int) -> List[UserOrganization]:
        """Get all members of an organization"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(UserOrganization)
                .options(selectinload(UserOrganization.user))
                .where(
                    and_(
                        UserOrganization.organization_id == organization_id,
                        UserOrganization.is_active == True
                    )
                )
                .order_by(UserOrganization.created_at)
            )
            return result.scalars().all()

    async def remove_user_from_organization(self, user_id: int, organization_id: int) -> bool:
        """Remove user from organization"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(UserOrganization).where(
                    and_(
                        UserOrganization.user_id == user_id,
                        UserOrganization.organization_id == organization_id
                    )
                )
            )
            user_org = result.scalar_one_or_none()
            if not user_org:
                return False

            user_org.is_active = False
            await db.commit()
            return True

    async def update_user_organization_role(self, user_id: int, organization_id: int, role: str) -> bool:
        """Update user role in organization"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(UserOrganization).where(
                    and_(
                        UserOrganization.user_id == user_id,
                        UserOrganization.organization_id == organization_id,
                        UserOrganization.is_active == True
                    )
                )
            )
            user_org = result.scalar_one_or_none()
            if not user_org:
                return False

            user_org.role = role
            await db.commit()
            return True

    async def count_organization_owners(self, organization_id: int) -> int:
        """Count number of owners in organization"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(UserOrganization).where(
                    and_(
                        UserOrganization.organization_id == organization_id,
                        UserOrganization.role == OrganizationRole.OWNER.value,
                        UserOrganization.is_active == True
                    )
                )
            )
            return len(result.scalars().all())

    async def count_organization_agents(self, organization_id: int) -> int:
        """Count number of active agents in organization"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(Agent).where(
                    and_(
                        Agent.organization_id == organization_id,
                        Agent.is_active == True
                    )
                )
            )
            return len(result.scalars().all())

    async def count_organization_users(self, organization_id: int) -> int:
        """Count number of active users in organization"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(UserOrganization).where(
                    and_(
                        UserOrganization.organization_id == organization_id,
                        UserOrganization.is_active == True
                    )
                )
            )
            return len(result.scalars().all())

    # Organization Invitation Operations
    async def create_organization_invitation(
        self,
        email: str,
        organization_id: int,
        role: str,
        invited_by_user_id: int,
        invitation_token: str,
        expires_at: datetime
    ) -> UserOrganization:
        """Create organization invitation"""
        async with await self.get_session() as db:
            # First check if user exists
            user_result = await db.execute(
                select(User).where(User.email == email)
            )
            user = user_result.scalar_one_or_none()

            if user:
                # Create invitation for existing user
                user_org = UserOrganization(
                    user_id=user.id,
                    organization_id=organization_id,
                    role=role,
                    invited_by_user_id=invited_by_user_id,
                    invitation_token=invitation_token,
                    invitation_expires_at=expires_at,
                    is_active=False  # Not active until accepted
                )
            else:
                # Create placeholder for non-existing user
                # We'll need to handle this case when they register
                user_org = UserOrganization(
                    user_id=None,  # Will be filled when user registers
                    organization_id=organization_id,
                    role=role,
                    invited_by_user_id=invited_by_user_id,
                    invitation_token=invitation_token,
                    invitation_expires_at=expires_at,
                    is_active=False
                )

            db.add(user_org)
            await db.commit()
            await db.refresh(user_org)
            return user_org

    async def get_organization_invitation_by_token(self, token: str) -> Optional[UserOrganization]:
        """Get organization invitation by token"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(UserOrganization)
                .options(selectinload(UserOrganization.user))
                .options(selectinload(UserOrganization.organization))
                .where(UserOrganization.invitation_token == token)
            )
            return result.scalar_one_or_none()

    async def accept_organization_invitation(self, invitation_id: int, user_id: int) -> bool:
        """Accept organization invitation"""
        async with await self.get_session() as db:
            result = await db.execute(
                select(UserOrganization).where(UserOrganization.id == invitation_id)
            )
            invitation = result.scalar_one_or_none()
            if not invitation:
                return False

            invitation.user_id = user_id
            invitation.is_active = True
            invitation.invitation_accepted_at = datetime.utcnow()
            invitation.invitation_token = None  # Clear token after acceptance

            await db.commit()
            return True


# Create a global instance
db_service = DatabaseService()


# Helper functions for backward compatibility
async def get_agent_by_id(agent_id: int) -> Optional[Agent]:
    """Get agent by ID - backward compatibility"""
    return await db_service.get_agent_by_id(agent_id)


async def create_demo_data():
    """Create some demo data for testing"""
    try:
        # Create demo user
        demo_user = await db_service.create_user(
            email="demo@aiagents.com",
            password_hash="demo_hash",
            name="Demo User",
            plan="pro"
        )

        # Create demo agents
        async with await get_db_session() as db:
            customer_support_agent = await db_service.create_agent(
                user_id=demo_user.id,
                name="Customer Support Bot",
                description="Helps customers with product questions and support",
                system_prompt="You are a helpful customer support assistant. Be friendly, professional, and helpful.",
                config={"temperature": 0.7, "max_tokens": 1000},
                widget_config={"theme": "blue", "position": "bottom-right"}
            )

            sales_agent = await db_service.create_agent(
                user_id=demo_user.id,
                name="Sales Assistant",
                description="Assists with product recommendations and sales inquiries",
                system_prompt="You are a knowledgeable sales assistant. Help customers find the right products.",
                config={"temperature": 0.8, "max_tokens": 1200},
                widget_config={"theme": "green", "position": "bottom-left"}
            )
            await db.commit()

        # Create demo documents
        await db_service.create_document(
            agent_id=customer_support_agent.id,
            filename="faq.txt",
            content="Frequently Asked Questions:\n\nQ: What is our return policy?\nA: We offer 30-day returns on all products.\n\nQ: How do I contact support?\nA: You can reach us at support@company.com or through this chat.",
            content_type="text/plain",
            doc_metadata={"source": "support_docs", "category": "faq"}
        )

        await db_service.create_document(
            agent_id=sales_agent.id,
            filename="product_catalog.txt",
            content="Product Catalog:\n\n1. Premium Plan - $99/month\n   - Unlimited agents\n   - Priority support\n   - Advanced analytics\n\n2. Basic Plan - $29/month\n   - 5 agents\n   - Standard support\n   - Basic analytics",
            content_type="text/plain",
            doc_metadata={"source": "sales_docs", "category": "pricing"}
        )

        print("✅ Demo data created successfully!")
        return {
            "user": demo_user,
            "agents": [customer_support_agent, sales_agent]
        }

    except Exception as e:
        print(f"❌ Error creating demo data: {e}")
        return None


if __name__ == "__main__":
    import asyncio

    async def test_database():
        """Test database operations"""
        print("🧪 Testing database service...")

        # Create demo data
        demo_data = await create_demo_data()

        if demo_data:
            # Test retrieving data
            user = demo_data["user"]
            agents = await db_service.get_user_agents(user.id)

            print(f"📊 User: {user.name} ({user.email})")
            print(f"🤖 Agents: {len(agents)}")

            for agent in agents:
                documents = await db_service.get_agent_documents(agent.id)
                stats = await db_service.get_agent_stats(agent.id)
                print(f"   - {agent.name}: {len(documents)} documents, {stats['total_conversations']} conversations")

            # Test system stats
            system_stats = await db_service.get_system_stats()
            print(f"🌍 System stats: {system_stats}")

        print("✅ Database service test completed!")

    asyncio.run(test_database())
