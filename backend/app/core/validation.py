"""
Comprehensive input validation for production-ready API.
"""

import re
import uuid
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from email_validator import validate_email, EmailNotValidError

from .exceptions import ValidationException

class BaseValidator(BaseModel):
    """Base validator with common validation methods"""
    
    @classmethod
    def validate_email_format(cls, email: str) -> str:
        """Validate email format"""
        if not email:
            raise ValueError("Email is required")
        
        try:
            validated_email = validate_email(email)
            return validated_email.email
        except EmailNotValidError as e:
            raise ValueError(f"Invalid email format: {str(e)}")
    
    @classmethod
    def validate_uuid_format(cls, uuid_str: str, field_name: str = "UUID") -> str:
        """Validate UUID format"""
        if not uuid_str:
            raise ValueError(f"{field_name} is required")
        
        try:
            uuid.UUID(uuid_str)
            return uuid_str
        except ValueError:
            raise ValueError(f"Invalid {field_name} format")
    
    @classmethod
    def validate_slug_format(cls, slug: str, field_name: str = "Slug") -> str:
        """Validate slug format (alphanumeric, hyphens, underscores)"""
        if not slug:
            raise ValueError(f"{field_name} is required")
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', slug):
            raise ValueError(f"{field_name} can only contain letters, numbers, hyphens, and underscores")
        
        if len(slug) < 2 or len(slug) > 50:
            raise ValueError(f"{field_name} must be between 2 and 50 characters")
        
        return slug.lower()
    
    @classmethod
    def validate_name_format(cls, name: str, field_name: str = "Name") -> str:
        """Validate name format"""
        if not name:
            raise ValueError(f"{field_name} is required")

        name = name.strip()
        if len(name) < 1 or len(name) > 255:
            raise ValueError(f"{field_name} must be between 1 and 255 characters")
        
        # Check for potentially harmful characters
        if re.search(r'[<>"\']', name):
            raise ValueError(f"{field_name} contains invalid characters")

        return name

    @classmethod
    def validate_password_strength(cls, password: str) -> str:
        """Validate password strength"""
        if not password:
            raise ValueError("Password is required")
        
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        if len(password) > 128:
            raise ValueError("Password must be less than 128 characters")
        
        # Check for at least one uppercase, lowercase, digit, and special character
        if not re.search(r'[A-Z]', password):
            raise ValueError("Password must contain at least one uppercase letter")
        
        if not re.search(r'[a-z]', password):
            raise ValueError("Password must contain at least one lowercase letter")
        
        if not re.search(r'\d', password):
            raise ValueError("Password must contain at least one digit")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            raise ValueError("Password must contain at least one special character")
        
        return password

class AgentValidator(BaseValidator):
    """Validation for agent-related inputs"""
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    system_prompt: Optional[str] = Field(None, max_length=10000)
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        return cls.validate_name_format(v, "Agent name")
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        if v is not None:
            v = v.strip()
            if len(v) > 2000:
                raise ValueError("Description must be less than 2000 characters")
        return v
    
    @field_validator('system_prompt')
    @classmethod
    def validate_system_prompt(cls, v):
        if v is not None:
            v = v.strip()
            if len(v) > 10000:
                raise ValueError("System prompt must be less than 10000 characters")
        return v

class UserValidator(BaseValidator):
    """Validation for user-related inputs"""
    
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    name: str = Field(..., min_length=1, max_length=255)
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        return cls.validate_email_format(v)
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        return cls.validate_password_strength(v)
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        return cls.validate_name_format(v, "Name")

class OrganizationValidator(BaseValidator):
    """Validation for organization-related inputs"""
    
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=2, max_length=50)
    description: Optional[str] = Field(None, max_length=2000)
    website: Optional[str] = Field(None, max_length=255)
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        return cls.validate_name_format(v, "Organization name")
    
    @field_validator('slug')
    @classmethod
    def validate_slug(cls, v):
        return cls.validate_slug_format(v, "Organization slug")
    
    @field_validator('website')
    @classmethod
    def validate_website(cls, v):
        if v is not None:
            v = v.strip()
            if v and not re.match(r'^https?://', v):
                raise ValueError("Website must start with http:// or https://")
        return v

class DocumentValidator(BaseValidator):
    """Validation for document-related inputs"""
    
    title: str = Field(..., min_length=1, max_length=255)
    content: Optional[str] = Field(None, max_length=1000000)  # 1MB limit
    file_type: str = Field(..., pattern=r'^(pdf|txt|docx|md|html)$')
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        return cls.validate_name_format(v, "Document title")
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        if v is not None and len(v) > 1000000:
            raise ValueError("Document content must be less than 1MB")
        return v

class ChatValidator(BaseValidator):
    """Validation for chat-related inputs"""
    
    message: str = Field(..., min_length=1, max_length=4000)
    visitor_id: Optional[str] = Field(None, max_length=255)
    session_context: Optional[Dict[str, Any]] = Field(None)
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty or whitespace only")
        
        v = v.strip()
        if len(v) > 4000:
            raise ValueError("Message must be less than 4000 characters")
        
        # Check for potentially harmful content
        if re.search(r'<script|javascript:|data:|vbscript:', v, re.IGNORECASE):
            raise ValueError("Message contains potentially harmful content")
        
        return v
    
    @field_validator('visitor_id')
    @classmethod
    def validate_visitor_id(cls, v):
        if v is not None:
            v = v.strip()
            if not re.match(r'^[a-zA-Z0-9_-]+$', v):
                raise ValueError("Visitor ID can only contain letters, numbers, hyphens, and underscores")
        return v

class PaginationValidator(BaseValidator):
    """Validation for pagination parameters"""
    
    page: int = Field(1, ge=1, le=1000)
    per_page: int = Field(10, ge=1, le=100)
    
    @field_validator('per_page')
    @classmethod
    def validate_per_page(cls, v):
        if v > 100:
            raise ValueError("per_page cannot exceed 100")
        return v

class SearchValidator(BaseValidator):
    """Validation for search parameters"""
    
    query: str = Field(..., min_length=1, max_length=255)
    filters: Optional[Dict[str, Any]] = Field(None)
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Search query cannot be empty")
        
        v = v.strip()
        if len(v) > 255:
            raise ValueError("Search query must be less than 255 characters")
        
        return v

class ConfigValidator(BaseValidator):
    """Validation for configuration objects"""
    
    @classmethod
    def validate_agent_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate agent configuration"""
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
        
        # Validate model field
        if 'model' in config:
            allowed_models = ['gpt-4', 'gpt-3.5-turbo', 'gemini-2.0-flash-exp', 'claude-3-sonnet']
            if config['model'] not in allowed_models:
                raise ValueError(f"Model must be one of: {', '.join(allowed_models)}")
        
        # Validate temperature
        if 'temperature' in config:
            temp = config['temperature']
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                raise ValueError("Temperature must be a number between 0 and 2")
        
        # Validate max_tokens
        if 'max_tokens' in config:
            tokens = config['max_tokens']
            if not isinstance(tokens, int) or tokens < 1 or tokens > 4000:
                raise ValueError("Max tokens must be an integer between 1 and 4000")
        
        return config
    
    @classmethod
    def validate_widget_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate widget configuration"""
        if not isinstance(config, dict):
            raise ValueError("Widget config must be a dictionary")
        
        # Validate theme
        if 'theme' in config:
            allowed_themes = ['modern', 'classic', 'minimal', 'dark']
            if config['theme'] not in allowed_themes:
                raise ValueError(f"Theme must be one of: {', '.join(allowed_themes)}")
        
        # Validate position
        if 'position' in config:
            allowed_positions = ['bottom-right', 'bottom-left', 'top-right', 'top-left']
            if config['position'] not in allowed_positions:
                raise ValueError(f"Position must be one of: {', '.join(allowed_positions)}")
        
        # Validate size
        if 'size' in config:
            allowed_sizes = ['small', 'medium', 'large']
            if config['size'] not in allowed_sizes:
                raise ValueError(f"Size must be one of: {', '.join(allowed_sizes)}")
        
        return config

def validate_input(data: Any, validator_class: type) -> Any:
    """Validate input data using a validator class"""
    try:
        if isinstance(data, dict):
            return validator_class(**data)
        elif isinstance(data, validator_class):
            return data
        else:
            raise ValueError(f"Invalid data type for {validator_class.__name__}")
    except Exception as e:
        raise ValidationException(f"Validation failed: {str(e)}")

def validate_list_input(data: List[Any], validator_class: type) -> List[Any]:
    """Validate a list of input data"""
    if not isinstance(data, list):
        raise ValidationException("Input must be a list")
    
    validated_items = []
    for i, item in enumerate(data):
        try:
            validated_items.append(validate_input(item, validator_class))
        except Exception as e:
            raise ValidationException(f"Validation failed for item {i}: {str(e)}")
    
    return validated_items