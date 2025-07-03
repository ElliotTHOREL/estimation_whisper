import services.services_translate as services_translate
from fastapi import APIRouter, Request  



router = APIRouter(prefix="/translate", tags=["Translate"])

@router.get("/one")
async def translate_one(model, id_audio, request: Request):
    return services_translate.translate_one(request.app, model, id_audio)

@router.post("/all")
async def translate_all(request: Request, replace = True):
    """On remplit la table audio_model_results avec :
    - tous les modèles montés
    - tous les audios de la base de données
    """
    services_translate.translate_all(request.app, replace)
