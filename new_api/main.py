import uvicorn
import os

from app import create_app
from controllers import create_tables, register_routes
from huggingface_hub import login

# Création de l'app
app = create_app()

# Connexion à Hugging Face
login(token=os.getenv("TOKEN_HF"))

# Création des tables
create_tables()

# Enregistrement des routes
register_routes(app)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=3344,
        reload=False  # Pour le développement
    )