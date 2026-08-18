"""Application configuration settings management using Pydantic Settings."""

import os
from typing import List, Union
from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Explicitly load .env files into os.environ
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")))



class Settings(BaseSettings):
    PROJECT_NAME: str = "HH Goa Voice RAG"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # API Server
    API_V1_STR: str = "/api"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS origins
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Future Service Credentials (Preserving placeholders)
    SARVAM_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    # Vector & Keyword Index Placeholders
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "hh_goa_chunks"
    BM25_INDEX_PATH: str = "data/bm25_index/bm25.pkl"

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
