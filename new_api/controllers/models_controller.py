import services.models as services_models
from fastapi import APIRouter, Request  

router = APIRouter(prefix="/models", tags=["Models"])

#CREATE
@router.post("/load")
async def load_model(model, request: Request):
    services_models.load_model(request.app,model)

#READ
@router.get("/")
async def get_models(request: Request):
    loaded_models = services_models.get_all_active_models(request.app)
    
    return ({
        "models": [
            {"name": model[0], "details": model[1]} 
            for model in loaded_models
        ],
        "count": len(loaded_models)
    })

#DELETE
@router.delete("/unload")
async def unload_model(model, request: Request):
    services_models.unload_model(request.app, model)

@router.delete("/clear")
async def clear_models(request: Request):
    services_models.clear_models(request.app)