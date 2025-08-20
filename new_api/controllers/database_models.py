import services.database.models as services_modeles_database
from fastapi import APIRouter

router = APIRouter(prefix="/modeles_database", tags=["Database modèles"])



#READ
@router.get("/all_names")
async def get_all_model_names():
    return (services_modeles_database.get_all_model_names())

@router.get("/all_details")
async def get_all_models():
    return (services_modeles_database.get_all_models())

@router.get("/types_valides")
async def get_types_valides():
    return (services_modeles_database.get_types_valides())

#UPDATE
@router.post("/add_one")
async def load_model(model, vrai_modele, type_modele, sampling_rate):
    services_modeles_database.ajoute_model(model, vrai_modele, type_modele, sampling_rate)

@router.post("/add_all_base_models")
async def add_all_base_models():
    services_modeles_database.add_all_base_models()

#DELETE
@router.delete("/one")
async def delete_model(model:str):
    services_modeles_database.delete_model(model)

@router.delete("/all")
async def reset_models():
    services_modeles_database.reset_models()






