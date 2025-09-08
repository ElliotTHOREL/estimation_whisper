from services.database.models import get_all_model_names, find_type_modele, find_vrai_modele

import torch
import logging
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Wav2Vec2Processor, Wav2Vec2ForCTC
from speechbrain.inference import EncoderASR, EncoderDecoderASR

from datasets import load_dataset
from dotenv import load_dotenv
import os
import mysql.connector
import gc
import time





load_dotenv()  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




device_str = os.getenv("TORCH_DEVICE", "cpu")

try:
    device = torch.device(device_str)
except Exception as e:
    logging.warning(f"Impossible d'utiliser le device '{device_str}' : {e}. Utilisation du CPU à la place.")
    device = torch.device("cpu")

AVAILABLE_MODELS = get_all_model_names()

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
    
    type_modele = find_type_modele(model)
    vrai_modele = find_vrai_modele(model)

    start_time = time.perf_counter()

    if type_modele in  ("whisper", "kyutai", "seamless"):
        processor = AutoProcessor.from_pretrained(vrai_modele)
        modele = AutoModelForSpeechSeq2Seq.from_pretrained(vrai_modele).to(device)

        app.state.models[model] = {"processor": processor, "model": modele}

    elif type_modele == "wav2vec2":
        processor = Wav2Vec2Processor.from_pretrained(vrai_modele)
        modele = Wav2Vec2ForCTC.from_pretrained(vrai_modele).to(device)

        app.state.models[model] = {"processor": processor, "model": modele}

    elif type_modele in ("speechbrain_seq2seq", "speechbrain_ctc"):
        modele = EncoderDecoderASR.from_hparams(source=vrai_modele, savedir=f"pretrained_models/{model}", run_opts={"device":device})

        app.state.models[model] = {"model": modele}

    else:
        raise ValueError(f"Le modèle {model} n'est pas supporté")


    _ = modele.eval() #Forcer le chargement complet du modèle  

    end_time = time.perf_counter()
    duree_chargement = end_time - start_time
    logging.info(f"Model {model} loaded in {duree_chargement:.2f}s on {device}")





#READ
def get_all_active_models(app):
    loaded_models=[]
    for model in app.state.models.keys():
        loaded_models.append((model, vrai_modele(model)))
    return loaded_models


#UPDATE -> pas d'update

#DELETE

def unload_model(app, model):



    if model in app.state.models:

        if 'processor' in app.state.models[model]:
            del app.state.models[model]['processor']
        if 'model' in app.state.models[model]:
            del app.state.models[model]['model']


        del app.state.models[model]
        logging.info(f"Le modèle {model} a été déchargé")
    else:
        logging.info(f"Teantative d'unload : Le modèle {model} n'est pas chargé")
    

    gc.collect()  # Au cas où il y aurait des références circulaires
    
    # Pour GPU (si utilisé)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()





def clear_models(app):
    for model in list(app.state.models.keys()):
        unload_model(app, model)
    
    
    gc.collect()  # Au cas où il y aurait des références circulaires
    

    app.state.models.clear()
    gc.collect()

    # Pour GPU (si utilisé)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()




