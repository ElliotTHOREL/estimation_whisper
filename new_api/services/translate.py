from transformers import pipeline

from services.models import sanity_check_models, get_all_active_models
from services.database.audio import get_all_audio, get_audio_path
from services.database.results import ajoute_result
from services.database.models import ajoute_model

import os
import librosa

import time



def get_full_path(id_audio):
    path_dataset = os.getenv("PATH_DATASET")
    path_spe_audio = get_audio_path(id_audio)
    return os.path.join(path_dataset, "clips",path_spe_audio)

def translate_one(app, nom_model, id_audio):
    sanity_check_models(app)
    
    start_time = time.time()
    if nom_model in ["w-tiny", "w-base", "w-small", "w-medium", "w-large-v2", "w-large-v3"]:
        transcription = translate_one_with_whisper(app, nom_model, id_audio)
    
    end_time = time.time()
    return transcription, end_time - start_time

def translate_one_with_whisper(app, nom_model, id_audio):
    path_audio = get_full_path(id_audio)

    if nom_model not in app.state.models:
        raise ValueError(f"Le modèle {nom_model} n'est pas chargé")
    
    processor = app.state.models[nom_model]["processor"]
    model = app.state.models[nom_model]["model"]

    audio_data, sampling_rate = librosa.load(path_audio, sr=16000)
    inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt")

    generate_kwargs = {}
    generate_kwargs["language"] = "fr"
    
    predicted_ids = model.generate(**inputs, **generate_kwargs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription[0]





