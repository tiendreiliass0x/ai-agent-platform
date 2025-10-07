"""Celery task package initialization."""

from .document_tasks import process_document, process_webpage
from .crawl_tasks import discover_urls

__all__ = [
    "process_document",
    "process_webpage",
    "discover_urls",
]
