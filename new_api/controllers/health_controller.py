from datetime import datetime
from fastapi import APIRouter

from services.database.results import reset_results
from services.database.results_model import reset_results_model
from services.database.batch_audio import reset_batch_audio
from services.database.audio import reset_audio
from services.database.models import reset_models

router = APIRouter()

@router.get("/health", tags=["Health"])
async def health_check():
    """Vérification de santé de l'API"""
    return {
        "status": "ok",
        "version": "2.0.0",
        "timestamp": datetime.now()
    }

@router.delete("/reset_tables", tags=["Reset tables"])
async def reset_tables():
    reset_results()
    reset_results_model()
    reset_batch_audio()
    reset_audio()
    reset_models()
