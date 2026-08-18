"""FastAPI Application Entrypoint for HH Goa 2026 Voice RAG System."""

import os
import sys
from contextlib import asynccontextmanager

# Ensure project root directory is in sys.path for top-level module imports (orchestration, voice, etc.)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import settings
from app.core.logging import logger, setup_logging

# Populate os.environ with configured API keys so child services can access them
if settings.SARVAM_API_KEY and not os.getenv("SARVAM_API_KEY"):
    os.environ["SARVAM_API_KEY"] = settings.SARVAM_API_KEY
if settings.GEMINI_API_KEY and not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager handling startup and shutdown events."""
    setup_logging()
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT}]")
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# CORS Configuration for Frontend local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static demo directory if available
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/demo", StaticFiles(directory=static_dir, html=True), name="static")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global unhandled exception handler ensuring clean JSON error responses."""
    logger.error(f"Unhandled error processing {request.method} {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected server error occurred.",
            },
        },
    )


# Include API Router under /api
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", include_in_schema=False)
async def root():
    """Root redirect to health check endpoint."""
    return {"message": f"Welcome to {settings.PROJECT_NAME}", "health": f"{settings.API_V1_STR}/health", "demo": "/demo"}
