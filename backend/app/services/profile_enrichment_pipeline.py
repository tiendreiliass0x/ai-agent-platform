"""
Real-Time Profile Enrichment Pipeline
Continuously enriches user profiles with intelligence from every interaction and touchpoint.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from app.services.user_intelligence_service import user_intelligence_service
from app.services.memory_service import memory_service
from app.services.gemini_service import gemini_service
from app.models.customer_profile import CustomerProfile
from app.models.customer_memory import MemoryType, MemoryImportance


class EnrichmentTrigger(str, Enum):
    """Events that trigger profile enrichment"""
    NEW_MESSAGE = "new_message"
    PAGE_VISIT = "page_visit"
    DOCUMENT_VIEW = "document_view"
    SESSION_END = "session_end"
    SATISFACTION_FEEDBACK = "satisfaction_feedback"
    BEHAVIOR_PATTERN_DETECTED = "behavior_pattern"
    PERIODIC_UPDATE = "periodic_update"


class EnrichmentPriority(str, Enum):
    """Priority levels for enrichment tasks"""
    CRITICAL = "critical"    # Immediate processing
    HIGH = "high"           # Process within 1 minute
    MEDIUM = "medium"       # Process within 5 minutes
    LOW = "low"            # Process when resources available


@dataclass
class EnrichmentTask:
    """Represents a profile enrichment task"""
    task_id: str
    customer_profile_id: int
    trigger: EnrichmentTrigger
    priority: EnrichmentPriority
    data: Dict[str, Any]
    context: Dict[str, Any]
    created_at: datetime
    scheduled_at: datetime
    retry_count: int = 0


@dataclass
class EnrichmentResult:
    """Result of profile enrichment"""
    task_id: str
    success: bool
    insights_discovered: List[Dict[str, Any]]
    profile_updates: Dict[str, Any]
    memory_entries: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    error: Optional[str] = None


class ProfileEnrichmentPipeline:
    """Orchestrates real-time profile enrichment across all user touchpoints"""

    def __init__(self):
        self.task_queue = asyncio.Queue()
        self.processing_tasks = {}
        self.enrichment_strategies = {
            EnrichmentTrigger.NEW_MESSAGE: self._enrich_from_message,
            EnrichmentTrigger.PAGE_VISIT: self._enrich_from_page_visit,
            EnrichmentTrigger.DOCUMENT_VIEW: self._enrich_from_document_view,
            EnrichmentTrigger.SESSION_END: self._enrich_from_session,
            EnrichmentTrigger.SATISFACTION_FEEDBACK: self._enrich_from_feedback,
            EnrichmentTrigger.BEHAVIOR_PATTERN_DETECTED: self._enrich_from_behavior,
            EnrichmentTrigger.PERIODIC_UPDATE: self._enrich_periodic_update
        }

    async def start_pipeline(self):
        """Start the enrichment pipeline"""
        # Start multiple workers for parallel processing
        workers = [
            asyncio.create_task(self._enrichment_worker(f"worker_{i}"))
            for i in range(3)  # 3 concurrent workers
        ]

        # Start periodic enrichment scheduler
        scheduler = asyncio.create_task(self._periodic_scheduler())

        await asyncio.gather(*workers, scheduler)

    async def enrich_profile(
        self,
        customer_profile_id: int,
        trigger: EnrichmentTrigger,
        data: Dict[str, Any],
        context: Dict[str, Any] = None,
        priority: EnrichmentPriority = EnrichmentPriority.MEDIUM
    ) -> str:
        """Queue a profile enrichment task"""

        task_id = f"enrich_{customer_profile_id}_{int(datetime.now().timestamp())}"

        task = EnrichmentTask(
            task_id=task_id,
            customer_profile_id=customer_profile_id,
            trigger=trigger,
            priority=priority,
            data=data,
            context=context or {},
            created_at=datetime.now(),
            scheduled_at=self._calculate_scheduled_time(priority)
        )

        await self.task_queue.put(task)
        return task_id

    async def _enrichment_worker(self, worker_id: str):
        """Worker that processes enrichment tasks"""

        while True:
            try:
                # Get next task from queue
                task = await self.task_queue.get()

                # Check if it's time to process
                if datetime.now() < task.scheduled_at:
                    # Put back in queue for later
                    await asyncio.sleep(1)
                    await self.task_queue.put(task)
                    continue

                # Process the task
                self.processing_tasks[task.task_id] = {
                    "worker": worker_id,
                    "started_at": datetime.now(),
                    "task": task
                }

                result = await self._process_enrichment_task(task)

                # Handle result
                await self._handle_enrichment_result(task, result)

                # Remove from processing
                if task.task_id in self.processing_tasks:
                    del self.processing_tasks[task.task_id]

                self.task_queue.task_done()

            except Exception as e:
                print(f"Error in enrichment worker {worker_id}: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    async def _process_enrichment_task(self, task: EnrichmentTask) -> EnrichmentResult:
        """Process a single enrichment task"""

        start_time = datetime.now()

        try:
            # Get the appropriate enrichment strategy
            strategy = self.enrichment_strategies.get(task.trigger)
            if not strategy:
                raise ValueError(f"No strategy for trigger: {task.trigger}")

            # Execute the enrichment strategy
            enrichment_data = await strategy(task)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            return EnrichmentResult(
                task_id=task.task_id,
                success=True,
                insights_discovered=enrichment_data.get("insights", []),
                profile_updates=enrichment_data.get("profile_updates", {}),
                memory_entries=enrichment_data.get("memory_entries", []),
                confidence_score=enrichment_data.get("confidence", 0.7),
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()

            return EnrichmentResult(
                task_id=task.task_id,
                success=False,
                insights_discovered=[],
                profile_updates={},
                memory_entries=[],
                confidence_score=0.0,
                processing_time=processing_time,
                error=str(e)
            )

    async def _enrich_from_message(self, task: EnrichmentTask) -> Dict[str, Any]:
        """Enrich profile from a new message interaction"""

        message = task.data.get("message", "")
        conversation_history = task.context.get("conversation_history", [])

        enrichment_data = {
            "insights": [],
            "profile_updates": {},
            "memory_entries": [],
            "confidence": 0.8
        }

        # Analyze message for insights
        analysis = await user_intelligence_service.analyze_user_message(
            message=message,
            customer_profile_id=task.customer_profile_id,
            conversation_history=conversation_history,
            session_context=task.context
        )

        # Extract insights
        enrichment_data["insights"].extend([
            {
                "type": "emotional_state",
                "value": analysis.emotional_state.value,
                "confidence": analysis.confidence_score,
                "source": "message_analysis",
                "timestamp": datetime.now().isoformat()
            },
            {
                "type": "intent",
                "value": analysis.intent_category.value,
                "confidence": analysis.confidence_score,
                "source": "message_analysis",
                "timestamp": datetime.now().isoformat()
            }
        ])

        # Update profile with communication patterns
        if analysis.personalization_cues:
            enrichment_data["profile_updates"]["communication_patterns"] = analysis.personalization_cues

        # Store key topics as memories
        for topic in analysis.key_topics:
            enrichment_data["memory_entries"].append({
                "memory_type": MemoryType.CONTEXTUAL.value,
                "key": f"topic_interest_{topic}",
                "value": f"User showed interest in {topic}",
                "importance": MemoryImportance.MEDIUM.value,
                "tags": ["topic", "interest"],
                "source": "message_analysis"
            })

        # Store pain points
        for pain_point in analysis.pain_points:
            enrichment_data["memory_entries"].append({
                "memory_type": MemoryType.EMOTIONAL.value,
                "key": f"pain_point_{datetime.now().strftime('%Y%m%d')}",
                "value": pain_point,
                "importance": MemoryImportance.HIGH.value,
                "tags": ["pain_point", "issue"],
                "source": "message_analysis"
            })

        # Advanced AI-powered insight extraction
        advanced_insights = await self._extract_advanced_message_insights(
            message, task.customer_profile_id
        )
        enrichment_data["insights"].extend(advanced_insights)

        return enrichment_data

    async def _enrich_from_page_visit(self, task: EnrichmentTask) -> Dict[str, Any]:
        """Enrich profile from page visit behavior"""

        page_url = task.data.get("page_url", "")
        time_on_page = task.data.get("time_on_page", 0)
        referrer = task.data.get("referrer", "")

        enrichment_data = {
            "insights": [],
            "profile_updates": {},
            "memory_entries": [],
            "confidence": 0.6
        }

        # Analyze page visit patterns
        page_insights = await self._analyze_page_visit_patterns(
            task.customer_profile_id, page_url, time_on_page
        )

        enrichment_data["insights"].extend(page_insights)

        # Infer interests from page content
        if "pricing" in page_url.lower():
            enrichment_data["memory_entries"].append({
                "memory_type": MemoryType.BEHAVIORAL.value,
                "key": "pricing_interest",
                "value": f"Visited pricing page, spent {time_on_page} seconds",
                "importance": MemoryImportance.HIGH.value,
                "tags": ["pricing", "purchase_intent"],
                "source": "page_visit"
            })

        if "documentation" in page_url.lower() or "api" in page_url.lower():
            enrichment_data["profile_updates"]["technical_level"] = "advanced"
            enrichment_data["memory_entries"].append({
                "memory_type": MemoryType.PREFERENCE.value,
                "key": "technical_documentation_preference",
                "value": "User prefers technical documentation",
                "importance": MemoryImportance.MEDIUM.value,
                "tags": ["technical", "documentation"],
                "source": "page_visit"
            })

        return enrichment_data

    async def _enrich_from_session(self, task: EnrichmentTask) -> Dict[str, Any]:
        """Enrich profile from complete session analysis"""

        session_data = task.data
        session_duration = session_data.get("duration", 0)
        pages_visited = session_data.get("pages_visited", [])
        interactions = session_data.get("interactions", [])

        enrichment_data = {
            "insights": [],
            "profile_updates": {},
            "memory_entries": [],
            "confidence": 0.85
        }

        # Session behavior analysis
        session_analysis = await self._analyze_session_behavior(
            task.customer_profile_id, session_data
        )

        enrichment_data["insights"].extend(session_analysis["insights"])

        # Update engagement metrics
        enrichment_data["profile_updates"]["avg_session_duration"] = session_duration
        enrichment_data["profile_updates"]["last_seen_at"] = datetime.now().isoformat()

        # Store session summary
        enrichment_data["memory_entries"].append({
            "memory_type": MemoryType.EPISODIC.value,
            "key": f"session_{datetime.now().strftime('%Y%m%d_%H%M')}",
            "value": json.dumps({
                "duration": session_duration,
                "pages": len(pages_visited),
                "interactions": len(interactions)
            }),
            "importance": MemoryImportance.LOW.value,
            "tags": ["session", "behavior"],
            "source": "session_analysis"
        })

        return enrichment_data

    async def _enrich_from_feedback(self, task: EnrichmentTask) -> Dict[str, Any]:
        """Enrich profile from user feedback"""

        rating = task.data.get("rating", 0)
        feedback_text = task.data.get("feedback", "")

        enrichment_data = {
            "insights": [],
            "profile_updates": {},
            "memory_entries": [],
            "confidence": 0.95  # Explicit feedback is high confidence
        }

        # Update satisfaction score
        enrichment_data["profile_updates"]["satisfaction_score"] = rating

        # Analyze feedback sentiment and content
        if feedback_text:
            feedback_insights = await self._analyze_feedback_content(
                feedback_text, task.customer_profile_id
            )
            enrichment_data["insights"].extend(feedback_insights)

        # Store feedback as high-importance memory
        enrichment_data["memory_entries"].append({
            "memory_type": MemoryType.EMOTIONAL.value,
            "key": f"feedback_{datetime.now().strftime('%Y%m%d')}",
            "value": f"Rating: {rating}/5. {feedback_text}",
            "importance": MemoryImportance.CRITICAL.value,
            "tags": ["feedback", "satisfaction"],
            "source": "user_feedback"
        })

        return enrichment_data

    async def _extract_advanced_message_insights(
        self, message: str, customer_profile_id: int
    ) -> List[Dict[str, Any]]:
        """Extract advanced insights using AI analysis"""

        try:
            insight_prompt = f"""
Analyze this user message for deep insights about their personality, expertise level, and needs:

Message: "{message}"

Extract insights about:
1. Technical expertise level (beginner, intermediate, expert)
2. Communication style (formal, casual, direct, detailed)
3. Personality traits (analytical, creative, practical, etc.)
4. Current emotional state beyond basic sentiment
5. Hidden needs or concerns not explicitly stated
6. Decision-making style (quick, analytical, consensus-driven)

Format as JSON with insights array containing objects with: type, value, confidence, reasoning
"""

            ai_response = await gemini_service.generate_response(
                prompt=insight_prompt,
                temperature=0.2,
                max_tokens=400
            )

            # Parse AI response (would need robust JSON parsing)
            insights = []
            # This is simplified - real implementation would parse JSON response
            insights.append({
                "type": "ai_analysis",
                "value": "Advanced user insights extracted",
                "confidence": 0.7,
                "source": "ai_analysis",
                "timestamp": datetime.now().isoformat()
            })

            return insights

        except Exception as e:
            print(f"Error in advanced insight extraction: {e}")
            return []

    async def _analyze_page_visit_patterns(
        self, customer_profile_id: int, page_url: str, time_on_page: int
    ) -> List[Dict[str, Any]]:
        """Analyze patterns in page visit behavior"""

        insights = []

        # High engagement pages
        if time_on_page > 120:  # More than 2 minutes
            insights.append({
                "type": "engagement",
                "value": "high_page_engagement",
                "confidence": 0.8,
                "source": "page_behavior",
                "details": {"page": page_url, "time": time_on_page}
            })

        # Quick bounces might indicate confusion or irrelevance
        if time_on_page < 10:
            insights.append({
                "type": "engagement",
                "value": "quick_bounce",
                "confidence": 0.7,
                "source": "page_behavior",
                "details": {"page": page_url, "time": time_on_page}
            })

        return insights

    async def _analyze_session_behavior(
        self, customer_profile_id: int, session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze complete session behavior patterns"""

        insights = []
        pages_visited = session_data.get("pages_visited", [])

        # Journey analysis
        if len(pages_visited) > 5:
            insights.append({
                "type": "behavior",
                "value": "thorough_explorer",
                "confidence": 0.8,
                "source": "session_analysis"
            })

        # Interest pattern analysis
        page_categories = self._categorize_pages(pages_visited)
        if page_categories:
            insights.append({
                "type": "interests",
                "value": page_categories,
                "confidence": 0.9,
                "source": "session_analysis"
            })

        return {"insights": insights}

    def _categorize_pages(self, pages: List[str]) -> Dict[str, int]:
        """Categorize visited pages to understand interests"""

        categories = {}
        for page in pages:
            if "pricing" in page.lower():
                categories["pricing"] = categories.get("pricing", 0) + 1
            elif "features" in page.lower():
                categories["features"] = categories.get("features", 0) + 1
            elif "docs" in page.lower() or "documentation" in page.lower():
                categories["documentation"] = categories.get("documentation", 0) + 1
            # Add more categorization logic

        return categories

    async def _analyze_feedback_content(
        self, feedback_text: str, customer_profile_id: int
    ) -> List[Dict[str, Any]]:
        """Analyze feedback content for insights"""

        insights = []

        # Sentiment analysis
        if any(word in feedback_text.lower() for word in ["excellent", "amazing", "love"]):
            insights.append({
                "type": "sentiment",
                "value": "highly_positive",
                "confidence": 0.9,
                "source": "feedback_analysis"
            })

        return insights

    async def _handle_enrichment_result(self, task: EnrichmentTask, result: EnrichmentResult):
        """Handle the result of an enrichment task"""

        if result.success:
            # Apply profile updates (would integrate with your database layer)
            await self._apply_profile_updates(
                task.customer_profile_id, result.profile_updates
            )

            # Store memory entries
            for memory_entry in result.memory_entries:
                await memory_service.store_memory(
                    customer_profile_id=task.customer_profile_id,
                    memory_type=MemoryType(memory_entry["memory_type"]),
                    key=memory_entry["key"],
                    value=memory_entry["value"],
                    importance=MemoryImportance(memory_entry["importance"]),
                    tags=memory_entry["tags"]
                )

            print(f"✅ Enrichment task {result.task_id} completed successfully")
            print(f"   Insights: {len(result.insights_discovered)}")
            print(f"   Profile updates: {len(result.profile_updates)}")
            print(f"   Memory entries: {len(result.memory_entries)}")

        else:
            # Handle failure
            if task.retry_count < 3:
                # Retry with exponential backoff
                task.retry_count += 1
                task.scheduled_at = datetime.now() + timedelta(minutes=2 ** task.retry_count)
                await self.task_queue.put(task)
                print(f"🔄 Retrying enrichment task {result.task_id} (attempt {task.retry_count})")
            else:
                print(f"❌ Enrichment task {result.task_id} failed permanently: {result.error}")

    async def _apply_profile_updates(self, customer_profile_id: int, updates: Dict[str, Any]):
        """Apply updates to customer profile"""
        # This would integrate with your database layer to update the CustomerProfile
        pass

    async def _periodic_scheduler(self):
        """Schedule periodic profile enrichment tasks"""

        while True:
            try:
                # Schedule periodic updates for active profiles
                await self._schedule_periodic_updates()
                await asyncio.sleep(300)  # Run every 5 minutes

            except Exception as e:
                print(f"Error in periodic scheduler: {e}")
                await asyncio.sleep(60)

    async def _schedule_periodic_updates(self):
        """Schedule periodic enrichment for profiles that need it"""
        # This would query your database for profiles that haven't been updated recently
        # and schedule periodic enrichment tasks
        pass

    def _calculate_scheduled_time(self, priority: EnrichmentPriority) -> datetime:
        """Calculate when to process a task based on priority"""

        now = datetime.now()
        if priority == EnrichmentPriority.CRITICAL:
            return now  # Immediate
        elif priority == EnrichmentPriority.HIGH:
            return now + timedelta(minutes=1)
        elif priority == EnrichmentPriority.MEDIUM:
            return now + timedelta(minutes=5)
        else:
            return now + timedelta(minutes=15)

    # Additional enrichment strategies
    async def _enrich_from_document_view(self, task: EnrichmentTask) -> Dict[str, Any]:
        """Enrich from document viewing behavior"""
        return {"insights": [], "profile_updates": {}, "memory_entries": [], "confidence": 0.6}

    async def _enrich_from_behavior(self, task: EnrichmentTask) -> Dict[str, Any]:
        """Enrich from detected behavior patterns"""
        return {"insights": [], "profile_updates": {}, "memory_entries": [], "confidence": 0.7}

    async def _enrich_periodic_update(self, task: EnrichmentTask) -> Dict[str, Any]:
        """Perform periodic profile enrichment"""
        return {"insights": [], "profile_updates": {}, "memory_entries": [], "confidence": 0.5}


# Global instance
profile_enrichment_pipeline = ProfileEnrichmentPipeline()