import services.database.results as services_results_database
from fastapi import APIRouter, Request


router = APIRouter(prefix="/results_database", tags=["Database results"])


#CREATE
@router.post("/many_audios")
async def translate_batch(request: Request, replace = True, batch = "nom du batch", deb=0, fin=10):
    """On remplit la table audio_model_results avec :
    - tous les modèles montés
    - tous les audios "sélectionnés"
    """
    services_results_database.translate_all_models_many_audios(request.app, replace, batch, int(deb), int(fin))

@router.post("/all")
async def translate_all(request: Request, replace = True):
    """On remplit la table audio_model_results avec :
    - tous les modèles montés
    - tous les audios de la base de données
    """
    services_results_database.translate_all_models_all_audios(request.app, replace)

@router.post("/wer")
async def estimer_wer_transcriptions():
    services_results_database.estimer_tous_les_wer()


#READ
@router.get("/")
async def get_results(id_audio: int, nom_batch: str, nom_model: str):
    return services_results_database.get_results(id_audio, nom_batch, nom_model)






#DELETE
@router.delete("/all")
async def reset_results():
    services_results_database.reset_results()
