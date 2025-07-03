import services.services_modeles_database as services_modeles_database
from fastapi import APIRouter
from services.services_models import AVAILABLE_MODELS

router = APIRouter(prefix="/modeles_database", tags=["Database modèles"])


#CREATE
@router.post("/load")
async def load_all():
    services_modeles_database.ajoute_model(AVAILABLE_MODELS)

#READ
@router.get("/")
async def get_all_models():
    return (services_modeles_database.get_all_models())

#DELETE
@router.delete("/")
async def delete_all_models():
    services_modeles_database.delete_all_models()



