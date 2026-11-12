"""API routers for ExAI-PanC v2."""

from backend.app.routers.auth import router as auth_router
from backend.app.routers.patients import router as patients_router
from backend.app.routers.clinical import router as clinical_router
from backend.app.routers.labs import router as labs_router
from backend.app.routers.imaging import router as imaging_router
from backend.app.routers.genomics import router as genomics_router
from backend.app.routers.predictions import router as predictions_router
from backend.app.routers.twin import router as twin_router
from backend.app.routers.simulation import router as simulation_router
from backend.app.routers.treatment import router as treatment_router
from backend.app.routers.report import router as report_router
from backend.app.routers.workflow import router as workflow_router

__all__ = [
    "auth_router",
    "patients_router",
    "clinical_router",
    "labs_router",
    "imaging_router",
    "genomics_router",
    "predictions_router",
    "twin_router",
    "simulation_router",
    "treatment_router",
    "report_router",
    "workflow_router",
]
