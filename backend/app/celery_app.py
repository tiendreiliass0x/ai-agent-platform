from celery import Celery
from celery.schedules import crontab
from .core.config import settings


celery_app = Celery(
    "ai_agent_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

# Enhanced configuration with queue routing and priorities
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    task_track_started=True,

    # Reliability settings
    task_acks_late=True,  # Tasks are acknowledged after completion, not before
    task_reject_on_worker_lost=True,  # Re-queue tasks if worker crashes

    # Retry settings
    task_default_retry_delay=60,  # Wait 60 seconds before retrying
    task_max_retries=3,  # Maximum 3 retries

    # Performance settings
    worker_prefetch_multiplier=4,  # Number of tasks to prefetch per worker
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks (prevent memory leaks)

    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_extended=True,  # Store additional task metadata

    # Queue routing - map tasks to specific queues
    task_routes={
        # High priority - user-facing operations (document processing)
        'app.tasks.document_tasks.process_document': {'queue': 'high_priority'},
        'app.tasks.document_tasks.process_webpage': {'queue': 'high_priority'},

        # Medium priority - background operations (will be added in later phases)
        # 'app.tasks.embedding_tasks.generate_embeddings': {'queue': 'medium_priority'},
        # 'app.tasks.analytics_tasks.compute_stats': {'queue': 'medium_priority'},

        # Low priority - batch/scheduled operations
        'app.tasks.crawl_tasks.discover_urls': {'queue': 'low_priority'},
    },

    # Default queue for unrouted tasks
    task_default_queue='default',
    task_default_priority=5,
)

# Beat schedule for periodic tasks (will be expanded in Phase 3)
celery_app.conf.beat_schedule = {
    # Placeholder for future scheduled tasks
    # 'daily-analytics': {
    #     'task': 'app.tasks.analytics_tasks.compute_daily_analytics',
    #     'schedule': crontab(hour=2, minute=0),  # 2 AM daily
    # },
}

# Import tasks explicitly to ensure they're registered
from app.tasks import document_tasks, crawl_tasks, context_engine_tasks  # noqa

# Discover tasks in app.tasks package
celery_app.autodiscover_tasks(["app.tasks"])  # type: ignore
