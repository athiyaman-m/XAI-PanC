"""
FastAPI application entry point for ExAI-PanC v2.
Production-ready Clinical Decision Support System.
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting ExAI-PanC v2 application...")
    try:
        from backend.app.database import init_db
        await init_db()
        logger.info("Database tables initialized")
    except Exception as e:
        logger.warning("Database initialization skipped: %s", e)
    yield
    logger.info("Shutting down ExAI-PanC v2 application...")


app = FastAPI(
    title="ExAI-PanC Clinical Decision Support Platform",
    description="AI-powered Clinical Digital Twin for Pancreatic Cancer — Research Use Only",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Large twin-mesh JSON payloads are gzip-compressed on the wire.
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=6)


@app.get("/", tags=["platform"])
async def root():
    return {
        "application": "ExAI-PanC Clinical Digital Twin Platform",
        "version": "2.0.0",
        "disclaimer": "Research use only. Not for diagnosis or treatment. Clinician review required.",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.get("/api/v1/health", tags=["platform"])
async def health():
    return {
        "status": "healthy",
        "service": "exaipanc-api",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Register All Routers ────────────────────────────────────────────────

from backend.app.routers.auth import router as auth_router
app.include_router(auth_router, prefix="/api/v1")

from backend.app.routers.patients import router as patients_router
app.include_router(patients_router, prefix="/api/v1")

from backend.app.routers.clinical import router as clinical_router
app.include_router(clinical_router, prefix="/api/v1")

from backend.app.routers.labs import router as labs_router
app.include_router(labs_router, prefix="/api/v1")

from backend.app.routers.imaging import router as imaging_router
app.include_router(imaging_router, prefix="/api/v1")

from backend.app.routers.genomics import router as genomics_router
app.include_router(genomics_router, prefix="/api/v1")

from backend.app.routers.predictions import router as predictions_router
app.include_router(predictions_router, prefix="/api/v1")

from backend.app.routers.twin import router as twin_router
app.include_router(twin_router, prefix="/api/v1")

from backend.app.routers.twin_meshes import router as twin_meshes_router
app.include_router(twin_meshes_router, prefix="/api/v1")

from backend.app.routers.simulation import router as simulation_router
app.include_router(simulation_router, prefix="/api/v1")

from backend.app.routers.treatment import router as treatment_router
app.include_router(treatment_router, prefix="/api/v1")

from backend.app.routers.report import router as report_router
app.include_router(report_router, prefix="/api/v1")

from backend.app.routers.workflow import router as workflow_router
app.include_router(workflow_router, prefix="/api/v1")


# ── Error Handler ───────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": request.url.path,
        },
    )


# ── Legacy endpoint (backward compatible) ──────────────────────────────

@app.post("/api/v1/predictions/run_legacy", tags=["legacy"])
async def run_prediction_legacy(request: Request):
    """Legacy prediction endpoint for backward compatibility."""
    import numpy as np
    from backend.prediction_engine import run_full_prediction

    def _serialize(v):
        if isinstance(v, dict):
            return {k: _serialize(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_serialize(x) for x in v]
        if isinstance(v, np.ndarray):
            return _serialize(v.tolist())
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v

    data = await request.json()
    inputs = {
        "age": data.get("age", 60),
        "sex": data.get("sex", "Male"),
        "ca199": float(data.get("ca199", 37)),
        "cea": float(data.get("cea", 3)),
        "bilirubin": float(data.get("bilirubin", 0.8)),
        "amylase": float(data.get("amylase", 95)),
        "weight_loss_pct": float(data.get("weight_loss_pct", 0)),
        "smoking": data.get("smoking", "Never"),
        "alcohol": data.get("alcohol", "None / Minimal"),
        "diabetes": data.get("diabetes", "None"),
        "family_history": bool(data.get("family_history", False)),
        "pancreatitis": bool(data.get("pancreatitis", False)),
        "genetic": bool(data.get("genetic", False)),
        "kras_expr": float(data.get("kras_expr", 5.0)),
        "tp53_expr": float(data.get("tp53_expr", 5.0)),
        "smad4_expr": float(data.get("smad4_expr", 5.0)),
        "erbb2_expr": float(data.get("erbb2_expr", 5.0)),
    }
    raw = run_full_prediction(inputs)
    return _serialize(raw)
