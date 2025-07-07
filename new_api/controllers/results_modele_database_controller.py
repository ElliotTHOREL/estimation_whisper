from fastapi import APIRouter, Request
import services.database.results_model as services_results_model_database


router = APIRouter(prefix="/results_modele_database", tags=["Results modèle database"])


#CREATE
@router.post("/")
async def ajoute_result_model(request: Request, model: str, nom_batch: str, taille_echantillon: int, replace: bool = False):
    services_results_model_database.ajoute_result_model(request.app, model, nom_batch, taille_echantillon, replace)




#READ  
@router.get("/")
async def get_all_results_model():
    return services_results_model_database.get_all_results_model()



#DELETE
@router.delete("/")
async def delete_results_model(id: int):
    services_results_model_database.delete_results_model(id)

@router.delete("/all")
async def reset_results_model():
    services_results_model_database.reset_results_model()