import services.database.batch_audio as services_batch_audio
from fastapi import APIRouter
import os

router = APIRouter(prefix="/batch_audio_database", tags=["Database batch audio"])


#CREATE
@router.post("/load")
async def load_batch_audio(name, path, path_fichier_metadonnees):
    services_batch_audio.add_batch_audio_extended(name, path, path_fichier_metadonnees)

#READ
@router.get("/")
async def get_all_batch_audio():
    return services_batch_audio.get_all_batch_audio()

#DELETE
@router.delete("/")
async def reset_batch_audio():
    services_batch_audio.reset_batch_audio()