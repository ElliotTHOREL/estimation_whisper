from dotenv import load_dotenv
import os
load_dotenv()
os.environ["HF_HOME"] = os.getenv("HF_HOME")


import uvicorn


from log_config import log_config
from app import create_app
from controllers import create_tables, register_routes
from huggingface_hub import login

# Connexion à Hugging Face
login(token=os.getenv("TOKEN_HF"))



# Création de l'app
app = create_app()

# Création des tables
create_tables()

# Enregistrement des routes
register_routes(app)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=int(os.getenv('PORT_API')),
        reload=False, # Pour le développement
        log_config=log_config
    )