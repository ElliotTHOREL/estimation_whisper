from datetime import datetime
from fastapi import APIRouter


router = APIRouter()

@router.get("/health", tags=["Health"])
async def health_check():
    """Vérification de santé de l'API"""
    return {
        "status": "ok",
        "version": "2.0.0",
        "timestamp": datetime.now()
    }
