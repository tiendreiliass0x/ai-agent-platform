# Queue System Analysis & Recommendations

**Date:** 2025-10-05
**Current Infrastructure:** Celery + Redis (partially implemented)

---

## Executive Summary

The application **already has Celery + Redis configured** but is only using it for **website crawling**. Many other long-running operations are currently handled synchronously or with FastAPI's `BackgroundTasks`, which can cause performance issues and scalability problems as the platform grows.

**Key Finding:** The infrastructure is in place; we just need to **expand queue usage** to cover all heavy operations.

---

## Current Queue Implementation

### ✅ What's Already Using Queues

**1. Website Crawling** (`app/tasks/crawl_tasks.py`)
- **Task:** `discover_urls` - URL discovery using Firecrawl/basic crawler
- **API:** `/api/v1/crawler/enqueue` + `/api/v1/crawler/{task_id}`
- **Status:** ✅ **Fully implemented with Celery**
- **Benefits:** Async crawling, progress tracking, result polling

**Configuration:**
```python
# app/celery_app.py
celery_app = Celery(
    "ai_agent_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)
```

---

## ⚠️ Operations Currently NOT Using Queues

### 1. **Document Processing** (HIGH PRIORITY)

**Current Implementation:** `BackgroundTasks` in FastAPI
- **File:** `app/api/v1/documents.py:41-105`
- **Function:** `process_document_background()`
- **Operations:**
  - File upload handling
  - PDF/document parsing (PyMuPDF)
  - Text chunking & semantic splitting
  - Embedding generation (via OpenAI/local models)
  - Vector storage (Pinecone/pgvector)
  - Metadata extraction
  - Keyword extraction (YAKE)

**Issues with Current Approach:**
- ❌ No retry mechanism for failed embeddings
- ❌ No progress tracking for large documents
- ❌ Can't scale horizontally (tied to API worker process)
- ❌ Memory-intensive operations block API workers
- ❌ No queue prioritization (urgent docs vs batch uploads)

**Recommendation:** **MIGRATE TO CELERY** ✅

**Estimated Time:** Processing 100-page PDF can take 30-120 seconds

---

### 2. **Embedding Generation** (HIGH PRIORITY)

**Current Implementation:** Direct async calls
- **File:** `app/services/embedding_service.py`
- **Operations:**
  - Batch embedding generation (sentence-transformers or OpenAI)
  - Can process 100s of chunks per document

**Issues:**
- ❌ API rate limits with OpenAI (expensive retries)
- ❌ No batching optimization across multiple documents
- ❌ GPU utilization not optimized (local models)
- ❌ Blocking operations during peak load

**Recommendation:** **IMPLEMENT CELERY TASK WITH BATCHING** ✅

**Estimated Time:** 500 chunks × 0.5s = 250 seconds for large doc

---

### 3. **Vector Store Operations** (MEDIUM PRIORITY)

**Current Implementation:** Direct async calls
- **File:** `app/services/vector_store.py`
- **Operations:**
  - Bulk vector insertion (Pinecone/pgvector)
  - Vector similarity search (cached)
  - Index updates

**Issues:**
- ❌ Bulk insertions can timeout
- ❌ No retry logic for network failures
- ❌ Pinecone rate limits can cause errors

**Recommendation:** **IMPLEMENT CELERY TASK FOR BULK OPERATIONS** ⚠️

---

### 4. **Analytics & Reporting** (MEDIUM PRIORITY)

**Current Implementation:** Synchronous database queries
- **File:** `app/services/database_service.py:646` (get_agent_stats)
- **File:** `app/services/database_service.py:910` (get_system_stats)

**Issues:**
- ⚠️ Complex aggregation queries can be slow
- ⚠️ No caching for frequently requested stats
- ⚠️ Blocking API response while calculating

**Recommendation:** **IMPLEMENT SCHEDULED CELERY TASKS** ⏰

**Example Use Cases:**
- Daily/hourly analytics aggregation
- Usage reports for organizations
- Conversation quality metrics
- Token usage tracking

---

### 5. **RAG Pipeline Operations** (LOW PRIORITY - Already Async)

**Current Implementation:** Async streaming responses
- **File:** `app/services/rag_service.py`
- **Operations:**
  - Vector search
  - Context reranking
  - LLM generation (streamed via SSE)

**Status:** ✅ **Acceptable as-is** (user-facing, needs real-time response)

**Note:** Keep using streaming for chat endpoints, but could queue **non-urgent** operations like:
- Context pre-computation
- Document pre-processing for known queries

---

### 6. **Agent Creation & Training** (LOW PRIORITY)

**Current Implementation:** Direct async
- **File:** `app/services/agent_creation_service.py`

**Operations:**
- Agent initialization
- Knowledge base setup
- Persona configuration

**Recommendation:** **Consider queuing for bulk agent creation** (e.g., organization onboarding)

---

### 7. **Profile Enrichment Pipeline** (MEDIUM PRIORITY)

**Current Implementation:** Direct async
- **File:** `app/services/profile_enrichment_pipeline.py`

**Operations:**
- Customer profile analysis
- Behavioral pattern detection
- Sentiment analysis

**Recommendation:** **QUEUE FOR BATCH PROCESSING** ⚠️

---

## Recommended Queue Architecture

### Queue Types (Celery Queues)

```python
# app/celery_app.py - Update configuration
celery_app.conf.update(
    task_routes={
        # High priority - user-facing operations
        'app.tasks.document_tasks.process_document': {'queue': 'high_priority'},
        'app.tasks.embedding_tasks.generate_embeddings': {'queue': 'high_priority'},

        # Medium priority - background operations
        'app.tasks.analytics_tasks.compute_stats': {'queue': 'medium_priority'},
        'app.tasks.profile_tasks.enrich_profile': {'queue': 'medium_priority'},

        # Low priority - batch/scheduled operations
        'app.tasks.crawl_tasks.discover_urls': {'queue': 'low_priority'},
        'app.tasks.analytics_tasks.daily_report': {'queue': 'scheduled'},
    },
    task_default_priority=5,
    task_acks_late=True,  # Re-queue failed tasks
    task_reject_on_worker_lost=True,
)
```

### Worker Deployment Strategy

```bash
# High-priority worker (more instances, less concurrency)
celery -A app.celery_app worker -Q high_priority -c 4 -n high@%h

# Medium-priority worker
celery -A app.celery_app worker -Q medium_priority -c 8 -n medium@%h

# Low-priority worker (fewer instances, more concurrency)
celery -A app.celery_app worker -Q low_priority -c 16 -n low@%h

# Scheduled task worker (Celery Beat)
celery -A app.celery_app beat --loglevel=info
```

---

## Implementation Roadmap

### Phase 1: Critical Operations (Week 1)

**1.1 Document Processing Task**
```python
# app/tasks/document_tasks.py
@celery_app.task(name="app.tasks.process_document", bind=True, max_retries=3)
def process_document(self, document_id: int, file_path: str, agent_id: int, ...):
    """Process uploaded document with retry logic"""
    try:
        # Move logic from process_document_background()
        # Add progress updates via self.update_state()
        pass
    except Exception as e:
        self.retry(exc=e, countdown=60)  # Retry after 1 min
```

**1.2 Update Document Upload Endpoint**
```python
# app/api/v1/documents.py
@router.post("/upload")
async def upload_document(...):
    # ... save file ...

    # Queue task instead of BackgroundTasks
    task = process_document.apply_async(
        args=[document_id, file_path, agent_id, ...],
        queue='high_priority'
    )

    return {"task_id": task.id, "status": "queued"}
```

**1.3 Add Task Status Endpoint**
```python
@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    result = AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.state,
        "progress": result.info.get('progress', 0) if isinstance(result.info, dict) else 0,
        "result": result.result if result.successful() else None
    }
```

**Estimated Implementation Time:** 2-3 days

---

### Phase 2: Embedding & Vector Operations (Week 2)

**2.1 Batch Embedding Task**
```python
# app/tasks/embedding_tasks.py
@celery_app.task(name="app.tasks.generate_embeddings", bind=True)
def generate_embeddings_batch(self, texts: List[str], model: str = "default"):
    """Generate embeddings with intelligent batching"""
    # Batch optimization for OpenAI API
    # GPU batching for local models
    # Progress tracking
    pass
```

**2.2 Vector Bulk Insert Task**
```python
@celery_app.task(name="app.tasks.bulk_insert_vectors", max_retries=5)
def bulk_insert_vectors(embeddings: List, metadata: List, namespace: str):
    """Bulk insert with Pinecone rate limit handling"""
    # Retry with exponential backoff
    # Batch size optimization
    pass
```

**Estimated Implementation Time:** 2-3 days

---

### Phase 3: Analytics & Scheduled Tasks (Week 3)

**3.1 Scheduled Analytics**
```python
# app/tasks/analytics_tasks.py
@celery_app.task(name="app.tasks.compute_daily_analytics")
def compute_daily_analytics():
    """Compute and cache daily analytics"""
    # Run complex aggregation queries
    # Cache results in Redis
    pass

# app/celery_app.py - Add beat schedule
celery_app.conf.beat_schedule = {
    'daily-analytics': {
        'task': 'app.tasks.compute_daily_analytics',
        'schedule': crontab(hour=2, minute=0),  # 2 AM daily
    },
    'hourly-stats': {
        'task': 'app.tasks.compute_hourly_stats',
        'schedule': crontab(minute=0),  # Every hour
    },
}
```

**3.2 Usage Report Generation**
```python
@celery_app.task(name="app.tasks.generate_usage_report")
def generate_usage_report(organization_id: int, period: str):
    """Generate detailed usage report for billing"""
    # Calculate token usage
    # Count API calls
    # Generate PDF/CSV report
    pass
```

**Estimated Implementation Time:** 2-3 days

---

### Phase 4: Profile Enrichment & Batch Operations (Week 4)

**4.1 Profile Enrichment Task**
```python
# app/tasks/profile_tasks.py
@celery_app.task(name="app.tasks.enrich_customer_profile")
def enrich_customer_profile(customer_id: int):
    """Enrich customer profile with ML insights"""
    # Sentiment analysis
    # Behavior pattern detection
    # Purchase propensity scoring
    pass
```

**4.2 Batch Operations**
```python
@celery_app.task(name="app.tasks.bulk_agent_creation")
def bulk_agent_creation(organization_id: int, agent_configs: List[Dict]):
    """Create multiple agents in batch"""
    for config in agent_configs:
        # Create agent
        # Setup knowledge base
        # Progress tracking
        pass
```

**Estimated Implementation Time:** 2-3 days

---

## Monitoring & Observability

### Celery Monitoring Tools

**1. Flower (Web UI)**
```bash
pip install flower
celery -A app.celery_app flower --port=5555
```
- Real-time task monitoring
- Worker status
- Task history
- Rate graphs

**2. Prometheus Metrics**
```python
# app/celery_app.py
from celery.signals import task_success, task_failure

@task_success.connect
def task_success_handler(sender=None, result=None, **kwargs):
    # Log metrics to Prometheus
    pass

@task_failure.connect
def task_failure_handler(sender=None, exception=None, **kwargs):
    # Alert on failures
    pass
```

**3. Logging Integration**
```python
celery_app.conf.update(
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s'
)
```

---

## Performance Impact Estimates

### Before Queue Implementation
- **Document Upload Response:** 30-120 seconds (blocking)
- **Concurrent Uploads:** Limited by API worker processes
- **Failed Processing:** Manual retry required
- **Peak Load Handling:** Poor (workers blocked)

### After Queue Implementation
- **Document Upload Response:** <1 second (immediate task ID)
- **Concurrent Uploads:** Unlimited (queue buffering)
- **Failed Processing:** Automatic retry (3 attempts)
- **Peak Load Handling:** Excellent (queue smoothing)

### Scalability Improvements
- **Horizontal Scaling:** Add more Celery workers independently
- **Resource Optimization:** GPU workers separate from CPU workers
- **Cost Efficiency:** Process during off-peak hours (scheduled tasks)

---

## Cost Considerations

### Infrastructure Additions Needed

**Redis** (already in use for rate limiting):
- Current usage: Rate limiting
- Additional usage: Task queue + result backend
- **Cost impact:** Minimal (existing Redis instance can handle it)

**Celery Workers:**
- **Option 1:** Run on existing API servers (no extra cost, limited scalability)
- **Option 2:** Dedicated worker instances
  - Small: 1 CPU, 2GB RAM → ~$10-20/month
  - Medium: 2 CPU, 4GB RAM → ~$40-80/month
  - Large: 4 CPU, 8GB RAM → ~$150-300/month

**Recommended Starting Point:**
- 1 high-priority worker (2 CPU, 4GB RAM)
- 1 medium-priority worker (2 CPU, 4GB RAM)
- **Total:** ~$80-160/month

**ROI:** Prevents API worker blocking, enables horizontal scaling, improves user experience

---

## Migration Checklist

### Pre-Migration
- [ ] Verify Redis is accessible and has sufficient memory
- [ ] Test Celery configuration in development
- [ ] Set up Flower monitoring
- [ ] Create task tracking database table (optional)

### Phase 1 Migration
- [ ] Implement document processing task
- [ ] Add task status endpoint
- [ ] Update document upload to use queue
- [ ] Deploy high-priority worker
- [ ] Monitor for 1 week

### Phase 2-4 Migration
- [ ] Implement embedding & vector tasks
- [ ] Add analytics scheduled tasks
- [ ] Deploy additional worker queues
- [ ] Set up alerting for task failures

### Post-Migration
- [ ] Document queue architecture in README
- [ ] Create runbook for worker management
- [ ] Set up automatic worker scaling (optional)
- [ ] Implement task archival/cleanup

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Redis failure | High - All queues down | Redis Sentinel HA setup, failover to in-memory queue |
| Worker crashes | Medium - Tasks delayed | Task acks_late=True, health checks, auto-restart |
| Task deadlocks | Medium - Queue blocked | Task timeouts, dead letter queue |
| Memory leaks | Low - Worker OOM | Periodic worker restarts (max_tasks_per_child) |

---

## Conclusion

**Current State:**
- ✅ Celery infrastructure exists but underutilized (only crawling)
- ⚠️ Critical operations (document processing, embeddings) use BackgroundTasks
- ❌ No scheduled tasks or batch operations

**Recommended Action:**
**Expand Celery usage** to cover all long-running operations, starting with document processing (highest impact).

**Timeline:** 4 weeks for full implementation
**Effort:** Medium (infrastructure exists, need task migration)
**ROI:** High (better UX, scalability, reliability)

**Next Steps:**
1. Review this analysis with the team
2. Prioritize phases based on business needs
3. Start with Phase 1 (document processing) - highest impact
4. Monitor results and adjust worker sizing
5. Expand to other operations incrementally
