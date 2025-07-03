import torch
import logging
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from dotenv import load_dotenv
import os
import mysql.connector
import gc

from services.services_database_modeles import create_table_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

AVAILABLE_MODELS = {
    "w-tiny": "openai/whisper-tiny",
    "w-base": "openai/whisper-base", 
    "w-small": "openai/whisper-small",
    "w-medium": "openai/whisper-medium",
    "w-large-v2": "openai/whisper-large-v2",
    "w-large-v3": "openai/whisper-large-v3"
}

#REMARQUE :
# app.state.models est un dictionnaire de dictionnaires
#Exemple : app.state.models ={ "w-tiny": {"processor": processor1, "model": model1},
#                               "w-base": {"processor": processor3, "model": model3}}
def sanity_check_models(app):
    """vérification de l'existence des briques fondamentales pour éviter les plantages"""
    if not hasattr(app.state, "models"):
        app.state.models = {}
    create_table_models()
    


#CREATE

def load_model(app, model):
    sanity_check_models(app)
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Modèle {model} non disponible")

    if model in app.state.models:
        logging.info(f"Le modèle {model} est déjà chargé")
        return
    
    if model in ["w-tiny", "w-base", "w-small", "w-medium", "w-large-v2", "w-large-v3"]:
        load_model_whisper(app, model)

def load_model_whisper(app, model):
    vrai_modele = AVAILABLE_MODELS[model]
    processor = AutoProcessor.from_pretrained(vrai_modele)
    modele = AutoModelForSpeechSeq2Seq.from_pretrained(vrai_modele)
    app.state.models[model] = {"processor": processor, "model": modele}

#READ

def get_all_active_models(app):
    sanity_check_models(app)
    loaded_models=[]
    for model in app.state.models.keys():
        loaded_models.append((model, AVAILABLE_MODELS[model]))
    print(loaded_models)
    return loaded_models


#UPDATE -> pas d'update

#DELETE

def unload_model(app, model):
    sanity_check_models(app)
    if model in app.state.models:
        del app.state.models[model]
    else:
        logging.info(f"Le modèle {model} n'est pas chargé")
    

    gc.collect()  # Au cas où il y aurait des références circulaires
    
    # Pour GPU (si utilisé)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def clear_models(app):
    sanity_check_models(app)
    for model in app.state.models.copy():
        del app.state.models[model]
    
    
    gc.collect()  # Au cas où il y aurait des références circulaires
    
    # Pour GPU (si utilisé)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()




