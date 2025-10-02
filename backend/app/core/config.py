from typing import List, Union
from pydantic_settings import BaseSettings
from pydantic import field_validator, model_validator
import os
import secrets

class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = False  # Secure default

    # API
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", secrets.token_urlsafe(64))
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(64))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours (reduced from 8 days)

    # Security
    JWT_ALGORITHM: str = "HS256"
    BCRYPT_ROUNDS: int = 12

    # CORS - Restrictive by default
    ALLOWED_ORIGINS: Union[List[str], str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # Database (optional for startup)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/ai_agent_platform"

    # AI Services
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = ""
    PINECONE_INDEX_NAME: str = "ai-agents"
    FIRECRAWL_API_KEY: str = ""

    # LangExtract
    LANGEXTRACT_API_KEY: str = ""

    # Redis (for caching and background tasks)
    REDIS_URL: str = "redis://localhost:6379"

    # File upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = "./uploads"

    @field_validator('ALLOWED_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @model_validator(mode="after")
    def enforce_secure_configuration(self) -> "Settings":
        env = (self.ENVIRONMENT or "development").lower()

        if env not in {"development", "dev", "test", "testing"}:
            missing = []
            if not os.getenv("SECRET_KEY"):
                missing.append("SECRET_KEY")
            if not os.getenv("JWT_SECRET_KEY"):
                missing.append("JWT_SECRET_KEY")
            if missing:
                raise ValueError(
                    f"Missing required secrets for {env} environment: {', '.join(missing)}"
                )

        return self

    model_config = {
        "env_file": ".env",
    }

settings = Settings()
