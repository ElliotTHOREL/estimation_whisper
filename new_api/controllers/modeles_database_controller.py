import services.database.models as services_modeles_database
from fastapi import APIRouter, Request
from services.models import AVAILABLE_MODELS

router = APIRouter(prefix="/modeles_database", tags=["Database modèles"])


#CREATE
@router.post("/load")
async def load_all():
    services_modeles_database.ajoute_model(AVAILABLE_MODELS)

#READ
@router.get("/")
async def get_all_models():
    return (services_modeles_database.get_all_models())


#UPDATE

@router.post("/calculate_wer_one")
async def compute_wer_un_modele(model):
    return (services_modeles_database.calculate_wer(model))

@router.post("/calculate_wer_all")
async def calculate_wer_tous_les_modeles(request: Request):
    return (services_modeles_database.calculate_wer_full(request.app))

#DELETE
@router.delete("/")
async def reset_models():
    services_modeles_database.reset_models()



