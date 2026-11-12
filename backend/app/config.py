from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "ExAI-PanC Clinical Decision Support Platform"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    REGULATORY_DISCLAIMER: str = (
        "Research use only. Clinical decision support — not FDA approved. "
        "Not a medical device. Clinician review required."
    )

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # SQLite keeps the local development setup self-contained. Production
    # deployments can override this with PostgreSQL through .env.
    DATABASE_URL: str = "sqlite+aiosqlite:///./backend/data/exai_panc.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "exaipanc"
    MINIO_SECRET_KEY: str = "exaipanc_dev_minio"
    MINIO_BUCKET: str = "exaipanc-imaging"
    MINIO_SECURE: bool = False

    JWT_SECRET_KEY: str = Field(default="dev-secret-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    AUTH_PROVIDER: str = "local"

    MODELS_DIR: str = str(Path(__file__).resolve().parents[2] / "models")
    UPLOAD_DIR: str = str(Path(__file__).resolve().parents[1] / "data" / "uploads")

    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on", "debug", "development", "dev"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
