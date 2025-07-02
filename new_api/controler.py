"""ceci est mon script de controller"""
from time import time
import services.py as services


app = ?


"""JE veux un check-health"""
@app.get("/health")
async def health_check():
    """Vérification de santé de l'API"""
    return {
        "status": "ok",
        "version": "2.0.0",
        "timestamp": time.time()
    }

@app.get("/transcribe")
async def transcribe(model, path_fichier_audio):
    services.transcribe(model, path_fichier_audio)

@app.post("/load_model")
async def load_model(model, language="fr"):
    services.load_model(model, language)


@app.post("/load_dataset"):
async def load_dataset(nb_audio=5):
    services.load_dataset(nb_audio)