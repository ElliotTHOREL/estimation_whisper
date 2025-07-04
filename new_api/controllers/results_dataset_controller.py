import services.database.results as services_results_database
from fastapi import APIRouter, Request


router = APIRouter(prefix="/results_database", tags=["Database results"])


#CREATE
@router.post("/all")
async def translate_all(request: Request, replace = True):
    """On remplit la table audio_model_results avec :
    - tous les modèles montés
    - tous les audios de la base de données
    """
    services_results_database.translate_all(request.app, replace)

@router.post("/wer")
async def estimer_wer_transcriptions():
    services_results_database.estimer_tous_les_wer()

#DELETE
@router.delete("/")
async def reset_results():
    services_results_database.reset_results()
