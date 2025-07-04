import uvicorn
from app import create_app
from controllers import create_tables, register_routes

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
        port=8000,
        reload=True  # Pour le développement
    )