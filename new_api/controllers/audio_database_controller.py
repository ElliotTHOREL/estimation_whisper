import services.database.audio as services_audio_database
from fastapi import APIRouter

router = APIRouter(prefix="/audio_database", tags=["Database audio"])



#READ
@router.get("/")
async def get_all_audio():
    return (services_audio_database.get_all_audio())

@router.get("/count")
async def get_number_of_audio():
    return(services_audio_database.get_number_of_audio())

#DELETE
@router.delete("/reset")
async def reset_dataset():
    services_audio_database.reset_dataset()