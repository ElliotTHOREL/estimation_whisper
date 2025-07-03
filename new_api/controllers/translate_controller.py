import services.services_translate as services_translate
from fastapi import APIRouter, Request  



router = APIRouter(prefix="/translate", tags=["Translate"])

@router.get("/one")
async def translate_one(model, id_audio, request: Request):
    return services_translate.translate_one(request.app, model, id_audio)[0]

