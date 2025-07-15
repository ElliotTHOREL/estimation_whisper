from transformers import pipeline

from services.models import  get_all_active_models
from services.database.audio import get_all_audio, get_audio_path
from services.database.batch_audio import get_batch_audio_path
from services.database.results import ajoute_result


import os
import librosa
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_full_path(id_audio, batch_audio):
    path_batch = get_batch_audio_path(batch_audio)
    path_spe_audio = get_audio_path(id_audio, batch_audio)
    return os.path.join(path_batch, path_spe_audio)

def translate_one(app, nom_model, id_audio, batch_audio):  
    if nom_model not in app.state.models:
        raise ValueError(f"Le modèle {nom_model} n'est pas chargé")

    if nom_model in ["w-tiny", "w-base", "w-small", "w-medium", "w-large-v2", "w-large-v3","b-w-large-v3","b-w-large-v3-distil","b-w-small-cv11","seamless-m4t-v2"]:
        transcription, duree = translate_one_with_whisper(app, nom_model, id_audio, batch_audio)
    elif nom_model in ["w2-b-960h", "w2-large", "b-w2", "b-w2-1b"]:
        transcription, duree = translate_one_with_wav2vec(app, nom_model, id_audio, batch_audio)
    elif nom_model in ["kyutai-1b"]:
        transcription, duree = translate_one_with_kyutai(app, nom_model, id_audio, batch_audio)

    
    return transcription, duree

def translate_one_with_whisper(app, nom_model, id_audio, batch_audio):
    path_audio = get_full_path(id_audio, batch_audio)

    processor = app.state.models[nom_model]["processor"]
    model = app.state.models[nom_model]["model"]

    audio_data, sampling_rate = librosa.load(path_audio, sr=16000)

    start_time = time.perf_counter()
    inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generate_kwargs = {}
    generate_kwargs["language"] = "fr"
    
    predicted_ids = model.generate(**inputs, **generate_kwargs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    end_time = time.perf_counter()

    duree = end_time - start_time

    return transcription[0], duree

def translate_one_with_wav2vec(app, nom_model, id_audio, batch_audio):
    path_audio = get_full_path(id_audio, batch_audio)

    processor = app.state.models[nom_model]["processor"]
    model = app.state.models[nom_model]["model"]

    audio_data, sampling_rate = librosa.load(path_audio, sr=16000)

    start_time = time.perf_counter()
    inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    end_time = time.perf_counter()

    duree = end_time - start_time

    return transcription, duree



def translate_one_with_kyutai(app, nom_model, id_audio, batch_audio):
    path_audio = get_full_path(id_audio, batch_audio)
    
    processor = app.state.models[nom_model]["processor"]
    model = app.state.models[nom_model]["model"]
    
    audio_data, sampling_rate = librosa.load(path_audio, sr=24000)

    start_time = time.perf_counter()
    inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    predicted_ids = model.generate(**inputs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    end_time = time.perf_counter()

    duree = end_time - start_time

    return transcription[0], duree