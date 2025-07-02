#!/usr/bin/env python3
"""
Script pour activer (précharger) une liste de modèles dans l'API Speech-to-Text universelle.

- Modifiez la liste MODELS_TO_ACTIVATE pour choisir les modèles à charger.
- Le script vérifie la santé de l'API, charge chaque modèle via l'endpoint /load_model, et affiche le statut.
"""

import requests
import time
import os
from typing import List
import torch
from speech_api import app
from pathlib import Path

# === CONFIGURATION ===
API_URL = os.environ.get("API_URL", "http://localhost:8000")
LANGUE = os.environ.get("LANGUE", "fr")

# Liste des modèles à activer (modifiable !)
# Format : (type, taille)
MODELS_TO_ACTIVATE: List[tuple] = [
    # Modèles Whisper (OpenAI)
    ("whisper", "tiny"),
    ("whisper", "base"),


]

# === FONCTIONS ===
def check_api_health(api_url: str) -> bool:
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        return r.status_code == 200
    except Exception as e:
        print(f"❌ Erreur lors de la vérification de l'API : {e}")
        return False

def activate_model(model_type: str, model_size: str, langue: str = "fr") -> bool:
    """
    Active un modèle via l'API en appelant l'endpoint /load_model
    """
    url = f"{API_URL}/load_model"
    data = {
        "model_type": model_type,
        "model_size": model_size,
        "language": langue
    }
    try:
        r = requests.post(url, data=data, timeout=1800)
        if r.status_code == 200:
            print(f"  ✅ Modèle {model_type}:{model_size} chargé avec succès.")
            return True
        else:
            print(f"  ❌ Erreur API ({r.status_code}): {r.text}")
            return False
    except Exception as e:
        print(f"  ❌ Exception lors de l'activation du modèle: {e}")
        return False

def main():
    print(f"\n=== Activation de modèles Speech-to-Text via l'API ===")
    print(f"API : {API_URL}")
    print(f"Langue : {LANGUE}")
    print(f"Modèles à activer : {MODELS_TO_ACTIVATE}")

    if not check_api_health(API_URL):
        print("❌ L'API n'est pas accessible. Lancez-la d'abord !")
        return

    for model_type, model_size in MODELS_TO_ACTIVATE:
        activate_model(model_type, model_size, LANGUE)
        time.sleep(2)  # Petite pause pour éviter de surcharger l'API

    print("\n✅ Tous les modèles demandés ont été traités.")

if __name__ == "__main__":
    main() 