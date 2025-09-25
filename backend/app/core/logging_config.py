"""
Production-ready logging configuration
"""

import logging
import sys
from typing import Dict, Any
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter for production"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""

        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, 'agent_id'):
            log_data['agent_id'] = record.agent_id
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'organization_id'):
            log_data['organization_id'] = record.organization_id
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(level: str = "INFO") -> None:
    """Setup production logging configuration"""

    # Create structured formatter
    formatter = StructuredFormatter()

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(console_handler)

    # Configure specific loggers
    logging.getLogger("uvicorn.access").disabled = True  # Disable uvicorn access logs
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)  # Reduce SQL noise


def get_logger(name: str) -> logging.Logger:
    """Get logger with consistent configuration"""
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin to add logging capabilities to classes"""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return logging.getLogger(self.__class__.__name__)

    def log_info(self, message: str, **kwargs):
        """Log info message with context"""
        self.logger.info(message, extra=kwargs)

    def log_error(self, message: str, **kwargs):
        """Log error message with context"""
        self.logger.error(message, extra=kwargs)

    def log_warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self.logger.warning(message, extra=kwargs)