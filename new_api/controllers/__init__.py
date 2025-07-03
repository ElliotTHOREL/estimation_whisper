"""
Importe tous les contrôleurs pour les enregistrer automatiquement
"""
from fastapi import FastAPI
from .health_controller import router as health_router
from .models_controller import router as models_router
from .database_controller import router as database_router
from .translate_controller import router as translate_router

def register_routes(app: FastAPI):
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(database_router)
    app.include_router(translate_router)
