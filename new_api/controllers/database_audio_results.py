import services.database.audio_results as services_results_database
from fastapi import APIRouter, Request


router = APIRouter(prefix="/database_audio_results", tags=["Database audio results"])


#CREATE

@router.post("/one")
async def translate_one(request: Request, model: str, id_audio: int, batch: str, replace = True):
    """On remplit la table pour 
    -un audio  sélectionné
    -un modèle sélectionné
    """
    services_results_database.translate_many_models_many_audios(request.app, [model], batch, id_audio, id_audio+1, replace)


@router.post("/many_audios")
async def translate_batch(request: Request, batch = "nom du batch", deb=0, fin=10, replace = True):
    """On remplit la table audio_model_results pour :
    - tous les modèles de la base de données
    - tous les audios "sélectionnés"
    """
    services_results_database.translate_all_models_many_audios(request.app, batch, int(deb), int(fin), replace)

@router.post("/all")
async def translate_all(request: Request, replace = True):
    """On remplit la table audio_model_results pour :
    - tous les modèles de la base de données
    - tous les audios de la base de données
    """
    services_results_database.translate_all_models_all_audios(request.app, replace)

@router.post("/wer")
async def estimer_wer_transcriptions():
    """On calcule les WER qui n'ont pas été déjà calculés"""
    services_results_database.estimer_tous_les_wer()


#READ
@router.get("/")
async def get_result(id_audio: int, nom_batch: str, nom_model: str):
    """On récupère le résultat pour un audio et un modèle donnés"""
    return services_results_database.get_results(id_audio, nom_batch, nom_model)






#DELETE
@router.delete("/delete_all")
async def reset_results():
    services_results_database.reset_results()
