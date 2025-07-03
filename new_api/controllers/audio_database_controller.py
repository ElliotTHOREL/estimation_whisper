import services.services_audio_database as services_audio_database
from fastapi import APIRouter

router = APIRouter(prefix="/audio_database", tags=["Database audio"])


#CREATE
@router.post("/load")
async def load_dataset(nb_audio=5):
    services_audio_database.load_dataset(nb_audio)

#READ
@router.get("/")
async def get_all_audio():
    return (services_audio_database.get_all_audio())

@router.get("/count")
async def get_number_of_audio():
    return(services_audio_database.get_number_of_audio())

#DELETE
@router.delete("/")
async def delete_dataset():
    services_audio_database.delete_dataset()