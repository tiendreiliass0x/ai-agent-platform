"""
Input validation and sanitization utilities.
Provides secure validation functions to prevent injection attacks and data corruption.
"""

import re
import html
import unicodedata
from typing import Optional, List, Union
from urllib.parse import urlparse
import email_validator
from pydantic import validator
import bleach
import logging

from .exceptions import SecurityError

logger = logging.getLogger(__name__)

# Regex patterns for common validations
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
UUID_PATTERN = re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$')
ALPHANUMERIC_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
SAFE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\s\-\.]{1,100}$')

# SQL injection detection patterns
SQL_INJECTION_PATTERNS = [
    re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)", re.IGNORECASE),
    re.compile(r"(--|#|/\*|\*/)", re.IGNORECASE),
    re.compile(r"('|('')|(\;)|(\\))", re.IGNORECASE),
]

# XSS detection patterns
XSS_PATTERNS = [
    re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"on\w+\s*=", re.IGNORECASE),
    re.compile(r"<iframe[^>]*>.*?</iframe>", re.IGNORECASE | re.DOTALL),
]

# Allowed HTML tags for sanitization
ALLOWED_HTML_TAGS = ['p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
ALLOWED_HTML_ATTRIBUTES = {'a': ['href', 'title'], '*': ['class']}


class InputValidator:
    """Comprehensive input validation and sanitization"""

    @staticmethod
    def validate_email(email: str) -> str:
        """Validate and sanitize email address"""
        if not email:
            raise SecurityError("Email cannot be empty")

        email = email.strip().lower()

        if len(email) > 254:  # RFC 5321 limit
            raise SecurityError("Email address too long")

        if not EMAIL_PATTERN.match(email):
            raise SecurityError("Invalid email format")

        try:
            # Use email-validator for comprehensive validation
            validated = email_validator.validate_email(email)
            return validated.email
        except email_validator.EmailNotValidError as e:
            raise SecurityError(f"Invalid email: {str(e)}")

    @staticmethod
    def validate_uuid(uuid_str: str) -> str:
        """Validate UUID format"""
        if not uuid_str:
            raise SecurityError("UUID cannot be empty")

        uuid_str = uuid_str.strip()

        if not UUID_PATTERN.match(uuid_str):
            raise SecurityError("Invalid UUID format")

        return uuid_str.lower()

    @staticmethod
    def validate_alphanumeric(value: str, field_name: str = "field") -> str:
        """Validate alphanumeric input with underscores and hyphens"""
        if not value:
            raise SecurityError(f"{field_name} cannot be empty")

        value = value.strip()

        if len(value) > 100:
            raise SecurityError(f"{field_name} too long (max 100 characters)")

        if not ALPHANUMERIC_PATTERN.match(value):
            raise SecurityError(f"{field_name} contains invalid characters")

        return value

    @staticmethod
    def validate_safe_name(name: str, field_name: str = "name") -> str:
        """Validate safe name input (letters, numbers, spaces, hyphens, underscores, dots)"""
        if not name:
            raise SecurityError(f"{field_name} cannot be empty")

        name = name.strip()

        if len(name) > 100:
            raise SecurityError(f"{field_name} too long (max 100 characters)")

        if not SAFE_NAME_PATTERN.match(name):
            raise SecurityError(f"{field_name} contains invalid characters")

        return name

    @staticmethod
    def validate_url(url: str) -> str:
        """Validate URL format and scheme"""
        if not url:
            raise SecurityError("URL cannot be empty")

        url = url.strip()

        if len(url) > 2048:  # Reasonable URL length limit
            raise SecurityError("URL too long")

        try:
            parsed = urlparse(url)
            if parsed.scheme not in ['http', 'https']:
                raise SecurityError("Invalid URL scheme (only http/https allowed)")

            if not parsed.netloc:
                raise SecurityError("Invalid URL format")

            return url
        except Exception as e:
            raise SecurityError(f"Invalid URL: {str(e)}")

    @staticmethod
    def detect_sql_injection(text: str) -> bool:
        """Detect potential SQL injection attempts"""
        if not text:
            return False

        for pattern in SQL_INJECTION_PATTERNS:
            if pattern.search(text):
                return True
        return False

    @staticmethod
    def detect_xss(text: str) -> bool:
        """Detect potential XSS attempts"""
        if not text:
            return False

        for pattern in XSS_PATTERNS:
            if pattern.search(text):
                return True
        return False

    @staticmethod
    def sanitize_text(text: str, max_length: int = 1000) -> str:
        """Sanitize text input"""
        if not text:
            return ""

        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)

        # Strip and limit length
        text = text.strip()[:max_length]

        # Check for injection attempts
        if InputValidator.detect_sql_injection(text):
            logger.warning(f"SQL injection attempt detected: {text[:100]}...")
            raise SecurityError("Potentially malicious input detected")

        if InputValidator.detect_xss(text):
            logger.warning(f"XSS attempt detected: {text[:100]}...")
            raise SecurityError("Potentially malicious input detected")

        # HTML escape for safety
        text = html.escape(text)

        return text

    @staticmethod
    def sanitize_html(html_content: str, max_length: int = 5000) -> str:
        """Sanitize HTML content using bleach"""
        if not html_content:
            return ""

        # Limit length
        html_content = html_content[:max_length]

        # Use bleach to sanitize HTML
        cleaned = bleach.clean(
            html_content,
            tags=ALLOWED_HTML_TAGS,
            attributes=ALLOWED_HTML_ATTRIBUTES,
            strip=True
        )

        return cleaned

    @staticmethod
    def validate_json_field(value: str, field_name: str = "field") -> str:
        """Validate JSON field values"""
        if not value:
            return ""

        value = value.strip()

        if len(value) > 10000:  # Limit JSON field size
            raise SecurityError(f"{field_name} too long")

        # Check for injection attempts
        if InputValidator.detect_sql_injection(value):
            raise SecurityError(f"Potentially malicious {field_name}")

        return value

    @staticmethod
    def validate_integer(value: Union[int, str], min_val: int = None, max_val: int = None) -> int:
        """Validate integer input with optional range"""
        if isinstance(value, str):
            if not value.strip().lstrip('-').isdigit():
                raise SecurityError("Invalid integer format")
            value = int(value.strip())
        elif not isinstance(value, int):
            raise SecurityError("Value must be an integer")

        if min_val is not None and value < min_val:
            raise SecurityError(f"Value must be at least {min_val}")

        if max_val is not None and value > max_val:
            raise SecurityError(f"Value must be at most {max_val}")

        return value

    @staticmethod
    def validate_list_length(items: List, max_length: int = 100, field_name: str = "list") -> List:
        """Validate list length"""
        if not isinstance(items, list):
            raise SecurityError(f"{field_name} must be a list")

        if len(items) > max_length:
            raise SecurityError(f"{field_name} too long (max {max_length} items)")

        return items


# Pydantic validator decorators for common validations
def email_validator_func(cls, v):
    """Pydantic validator for email fields"""
    return InputValidator.validate_email(v) if v else v


def uuid_validator_func(cls, v):
    """Pydantic validator for UUID fields"""
    return InputValidator.validate_uuid(v) if v else v


def safe_text_validator_func(max_length: int = 1000):
    """Pydantic validator factory for safe text fields"""
    def validator_func(cls, v):
        return InputValidator.sanitize_text(v, max_length) if v else v
    return validator_func


def safe_name_validator_func(cls, v):
    """Pydantic validator for safe name fields"""
    return InputValidator.validate_safe_name(v) if v else v


def url_validator_func(cls, v):
    """Pydantic validator for URL fields"""
    return InputValidator.validate_url(v) if v else v