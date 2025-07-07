from fastapi import FastAPI
from connection import initialize_pool

def create_app() -> FastAPI:
    initialize_pool()

    app = FastAPI(
        title="Mon API",
        description="API universelle pour la transcription de fichiers audio",
        version="2.0.0"
    )
    app.state.models = {}
    return app