import torch
import logging
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Wav2Vec2Processor, Wav2Vec2ForCTC


from datasets import load_dataset
from dotenv import load_dotenv
import os
import mysql.connector
import gc
import time

from services.database.models import ajoute_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AVAILABLE_MODELS = {
    "w-tiny": "openai/whisper-tiny",
    "w-base": "openai/whisper-base", 
    "w-small": "openai/whisper-small",
    "w-medium": "openai/whisper-medium",
    "w-large-v2": "openai/whisper-large-v2",
    "w-large-v3": "openai/whisper-large-v3",
    "b-w-large-v3": "bofenghuang/whisper-large-v3-french",
    "b-w-large-v3-distil": "bofenghuang/whisper-large-v3-french-distil-dec16",
    "b-w-small-cv11": "bofenghuang/whisper-small-cv11-french",
    "kyutai-1b": "kyutai/stt-1b-en_fr-trfs",

    "w2-b-960h": "facebook/wav2vec2-base-960h",
    "w2-large": "facebook/wav2vec2-large-xlsr-53-french",
    "b-w2": "bofenghuang/asr-wav2vec2-ctc-french",
    "b-w2-1b" :"bofenghuang/asr-wav2vec2-xls-r-1b-ctc-french",

    "seamless-m4t-v2": "facebook/seamless-m4t-v2-large",
}

#On essaiera
# - gemma (n'existe pas?)
# - kyutai (bcp d'install a priori)
# - seamless (petite galère dans mes souvenirs)

# RECO GROK
# - SpeechBrain
# - Vosk
# - NVIDIA NeMo Canary-1B
# - Reverb ASR (avec fine-tuning)
# - Julius
# - Picovoice Cheetah


#On se passera de nvidia qui est une galère monstre à installer
#REMARQUE :
# app.state.models est un dictionnaire de dictionnaires
#Exemple : app.state.models ={ "w-tiny": {"processor": processor1, "model": model1},
#                               "w-base": {"processor": processor3, "model": model3}}


#CREATE
def load_model(app, model):
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Modèle {model} non disponible")

    if model in app.state.models:
        logging.info(f"Le modèle {model} est déjà chargé")
        return
    
    if model in ["w-tiny", "w-base", "w-small", "w-medium", "w-large-v2", "w-large-v3","b-w-large-v3","b-w-large-v3-distil","b-w-small-cv11","kyutai-1b","seamless-m4t-v2"]:
        load_model_whisper(app, model)
    elif model in ["w2-b-960h","w2-large","b-w2","b-w2-1b"]:
        load_model_wav2vec(app, model)
    else:
        raise ValueError(f"Le modèle {model} n'est pas supporté")
        

def load_model_whisper(app, model):
    vrai_modele = AVAILABLE_MODELS[model]

    start_time = time.perf_counter()
    processor = AutoProcessor.from_pretrained(vrai_modele)
    modele = AutoModelForSpeechSeq2Seq.from_pretrained(vrai_modele, device_map="auto")
    _ = modele.eval() #Forcer le chargement complet du modèle  
    app.state.models[model] = {"processor": processor, "model": modele}
    end_time = time.perf_counter()
    duree_chargement = end_time - start_time
    ajoute_model(model, duree_chargement)

def load_model_wav2vec(app, model):
    vrai_modele = AVAILABLE_MODELS[model]

    start_time = time.perf_counter()    
    processor = Wav2Vec2Processor.from_pretrained(vrai_modele)
    modele = Wav2Vec2ForCTC.from_pretrained(vrai_modele)
    modele = modele.to(device)
    _ = modele.eval() #Forcer le chargement complet du modèle  
    app.state.models[model] = {"processor": processor, "model": modele}
    end_time = time.perf_counter()
    duree_chargement = end_time - start_time
    ajoute_model(model, duree_chargement)



#READ
def get_all_active_models(app):
    loaded_models=[]
    for model in app.state.models.keys():
        loaded_models.append((model, AVAILABLE_MODELS[model]))
    return loaded_models


#UPDATE -> pas d'update

#DELETE

def unload_model(app, model):
    if model in app.state.models:
        del app.state.models[model]
    else:
        logging.info(f"Le modèle {model} n'est pas chargé")
    

    gc.collect()  # Au cas où il y aurait des références circulaires
    
    # Pour GPU (si utilisé)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def clear_models(app):
    for model in app.state.models.copy():
        del app.state.models[model]
    
    
    gc.collect()  # Au cas où il y aurait des références circulaires
    
    # Pour GPU (si utilisé)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()




