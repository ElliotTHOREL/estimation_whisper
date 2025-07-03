from transformers import pipeline

from services.services_models import sanity_check_models, get_all_active_models
from services.services_database import get_all_audio, get_audio_path
from services.services_database_results import ajoute_result
from services.services_database_modeles import ajoute_model

import os
import librosa




def get_full_path(id_audio):
    path_dataset = os.getenv("PATH_DATASET")
    path_spe_audio = get_audio_path(id_audio)
    return os.path.join(path_dataset, "clips",path_spe_audio)

def translate_one(app, nom_model, id_audio):
    sanity_check_models(app)
    if nom_model in ["w-tiny", "w-base", "w-small", "w-medium", "w-large-v2", "w-large-v3"]:
        return translate_one_with_whisper(app, nom_model, id_audio)

def translate_one_with_whisper(app, nom_model, id_audio):
    path_audio = get_full_path(id_audio)

    if nom_model not in app.state.models:
        raise ValueError(f"Le modèle {nom_model} n'est pas chargé")
    
    processor = app.state.models[nom_model]["processor"]
    model = app.state.models[nom_model]["model"]

    audio_data, sampling_rate = librosa.load(path_audio, sr=16000)
    inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt")
    predicted_ids = model.generate(**inputs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)


    transcription = transcription[0]  #fastfix, je sais pas pk le result est initialment dans une liste
    return transcription



def translate_all(app, replace):
    liste_models = [nom for (nom, _) in get_all_active_models(app)]
    translate_many(app, liste_models, replace)

def translate_many(app, liste_models, replace):
    sanity_check_models(app)
    for model in liste_models:
        if model not in app.state.models.keys():
            raise ValueError(f"Le modèle {model} n'est pas chargé")
    ajoute_model(liste_models)


    for (id_audio, _ , _) in get_all_audio():
        for model in liste_models:
            transcription_result = translate_one(app, model, id_audio)
            ajoute_result(model, id_audio, transcription_result, replace)

