"""
Importe tous les contrôleurs pour les enregistrer automatiquement
"""
from fastapi import FastAPI
from .health import router as health_router
from .models import router as models_router
from .database_batch_audio import router as batch_audio_router
from .database_audio import router as database_router
from .translate import router as translate_router
from .database_audio_results import router as database_audio_results_router
from .database_models import router as database_models_router
from .database_models_results import router as database_models_results_router

from services.database.batch_audio import create_table_batch_audio
from services.database.audio import create_table_audio
from services.database.models import create_table_models
from services.database.audio_results import create_table_results
from services.database.models_results import create_table_results_model

def create_tables():
    create_table_batch_audio()
    create_table_audio()
    create_table_models()
    create_table_results()
    create_table_results_model()

def register_routes(app: FastAPI):
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(translate_router)
    app.include_router(batch_audio_router)
    app.include_router(database_router)
    app.include_router(database_models_router)
    app.include_router(database_audio_results_router)
    app.include_router(database_models_results_router)
