from celery import Celery
from .core.config import settings


celery_app = Celery(
    "ai_agent_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

# Minimal configuration; routes/queues can be expanded later
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    task_track_started=True,
)

# Discover tasks in app.tasks package
celery_app.autodiscover_tasks(["app.tasks"])  # type: ignore
