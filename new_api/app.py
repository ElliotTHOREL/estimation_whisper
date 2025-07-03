from fastapi import FastAPI

def create_app() -> FastAPI:
    app = FastAPI(
        title="Mon API",
        description="API universelle pour la transcription de fichiers audio",
        version="2.0.0"
    )
    return app