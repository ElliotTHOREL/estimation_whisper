import services.services_database as services_database
from fastapi import APIRouter

router = APIRouter(prefix="/database", tags=["Database"])


#CREATE
@router.post("/load")
async def load_dataset(nb_audio=5):
    services_database.load_dataset(nb_audio)

#READ
@router.get("/")
async def get_all_audio():
    return (services_database.get_all_audio())

@router.get("/count")
async def get_number_of_audio():
    return(services_database.get_number_of_audio())

#DELETE
@router.delete("/")
async def delete_dataset():
    services_database.delete_dataset()