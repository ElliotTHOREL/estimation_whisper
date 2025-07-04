"""
Importe tous les contrôleurs pour les enregistrer automatiquement
"""
from fastapi import FastAPI
from .health_controller import router as health_router
from .models_controller import router as models_router
from .batch_audio_database_controller import router as batch_audio_router
from .audio_database_controller import router as database_router
from .translate_controller import router as translate_router
from .results_dataset_controller import router as results_router
from .modeles_database_controller import router as modeles_router

from services.database.batch_audio import create_table_batch_audio
from services.database.audio import create_table_audio
from services.database.models import create_table_models
from services.database.results import create_table_results


def create_tables():
    create_table_batch_audio()
    create_table_audio()
    create_table_models()
    create_table_results()


def register_routes(app: FastAPI):
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(translate_router)
    app.include_router(batch_audio_router)
    app.include_router(database_router)
    app.include_router(modeles_router)
    app.include_router(results_router)
