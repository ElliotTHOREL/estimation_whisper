from services.audio_manager import  Audio_file, Chunk_audio
from services.models import load_model
from services.database.models import find_type_modele, find_sr_modele

import numpy as np
import torch
import time

import logging
from typing import List

import os
from dotenv import load_dotenv




load_dotenv()  


device_str = os.getenv("TORCH_DEVICE", "cpu")

try:
    device = torch.device(device_str)
except Exception as e:
    logging.warning(f"Impossible d'utiliser le device '{device_str}' : {e}. Utilisation du CPU à la place.")
    device = torch.device("cpu")




def translate_one (app, nom_model, id_audio, batch_audio, taille_batch=8):
    sr = find_sr_modele(nom_model)
    audio_file = Audio_file(id_audio, batch_audio,sr)
    audio_file.get_liste_chunks()
    transcriptions = []
    total_duration = 0.0

    batchs = [audio_file.liste_chunks[i:i + taille_batch] for i in range(0, len(audio_file.liste_chunks), taille_batch)]

    for batch in batchs:
        transcription, duree = translate_one_batch(app, nom_model, batch)
        transcriptions.extend(transcription)
        total_duration += duree

    full_transcription = " ".join(transcriptions)
    return full_transcription, total_duration






def translate_one_batch(app, nom_model, batch):  
    if nom_model not in app.state.models:
        logging.info(f"Le modèle {nom_model} n'est pas chargé. Chargement en cours...")
        load_model(app, nom_model)

    type_modele = find_type_modele(nom_model)
    if type_modele in ("whisper"):
        transcription, duree = translate_one_batch_with_whisper(app, nom_model, batch)
    elif type_modele == "wav2vec2":
        transcription, duree = translate_one_batch_with_wav2vec(app, nom_model,  batch)
    elif type_modele in ("kyutai"):
        transcription, duree = translate_one_batch_with_kyutai(app, nom_model,  batch)
    elif type_modele == "speechbrain_seq2seq":
        transcription, duree = translate_one_batch_with_speechbrain_seq2seq(app, nom_model,  batch)
    elif type_modele == "speechbrain_ctc":
        transcription, duree = translate_one_batch_with_speechbrain_ctc(app, nom_model,  batch)
    elif type_modele == "seamless":
        transcription, duree = translate_one_batch_with_seamless(app, nom_model,  batch)
    return transcription, duree

def translate_one_batch_with_whisper(app, nom_model, batch:list[Chunk_audio]):
    if len(batch) == 0:
        logging.warning("translate_one_batch_with_whisper: batch is empty")
        return [], 0

    processor = app.state.models[nom_model]["processor"]
    model = app.state.models[nom_model]["model"]



    liste_audio_data = [chunk.get_audio_data() for chunk in batch]
    sampling_rate = batch[0].sr



    start_time = time.perf_counter()
    inputs = processor(liste_audio_data, sampling_rate=sampling_rate, return_tensors="pt", attention_mask=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generate_kwargs = {}
    generate_kwargs["language"] = "fr"

    with torch.no_grad():
        predicted_ids = model.generate(**inputs, **generate_kwargs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    end_time = time.perf_counter()

    duree = end_time - start_time

    return transcription, duree

def translate_one_batch_with_wav2vec(app, nom_model, batch:list[Chunk_audio]):
    if len(batch) == 0:
        logging.warning("translate_one_batch_with_wav2vec: batch is empty")
        return [], 0

    processor = app.state.models[nom_model]["processor"]
    model = app.state.models[nom_model]["model"]


    liste_audio_data = [chunk.get_audio_data() for chunk in batch]
    sampling_rate = batch[0].sr


    start_time = time.perf_counter()
    inputs = processor(liste_audio_data, sampling_rate=sampling_rate, return_tensors="pt", attention_mask=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    end_time = time.perf_counter()

    duree = end_time - start_time

    return transcription, duree



def translate_one_batch_with_kyutai(app, nom_model, batch:list[Chunk_audio]):
    if len(batch) == 0:
        logging.warning("translate_one_batch_with_kyutai: batch is empty")
        return [], 0

    processor = app.state.models[nom_model]["processor"]
    model = app.state.models[nom_model]["model"]
    
    liste_audio_data = [chunk.get_audio_data() for chunk in batch]
    sampling_rate = batch[0].sr

    start_time = time.perf_counter()
    inputs = processor(liste_audio_data, sampling_rate=sampling_rate, return_tensors="pt", attention_mask=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        predicted_ids = model.generate(**inputs,)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    end_time = time.perf_counter()

    duree = end_time - start_time

    return transcription, duree



def translate_one_batch_with_speechbrain_seq2seq(app, nom_model, batch:list[Chunk_audio]):
    if len(batch) == 0:
        logging.warning("translate_one_batch_with_speechbrain: batch is empty")
        return [], 0


    
    model = app.state.models[nom_model]["model"]

    start_time = time.perf_counter()

    liste_audio_data = [chunk.get_audio_data() for chunk in batch]
    audio_array = np.array(liste_audio_data)
    waveform = torch.tensor(audio_array, dtype=torch.float32).to(device)
    wav_lengths = torch.tensor([waveform.shape[1]]).to(device)

    with torch.no_grad():
        transcription = model.transcribe_batch(waveform, wav_lengths)

    
    transcription = [transcription[0][i] for i in range(len(transcription[0]))]

    end_time = time.perf_counter()
    duree = end_time - start_time
    
    return transcription, duree


def translate_one_batch_with_speechbrain_ctc(app, nom_model, batch: list[Chunk_audio]):
    if len(batch) == 0:
        logging.warning("translate_one_batch_with_speechbrain_ctc: batch is empty")
        return [], 0

    model = app.state.models[nom_model]["model"]

    start_time = time.perf_counter()

    # Préparer les données audio
    liste_audio_data = [chunk.get_audio_data() for chunk in batch]
    audio_array = np.array(liste_audio_data)
    waveform = torch.tensor(audio_array, dtype=torch.float32).to(device)
    wav_lengths = torch.tensor([waveform.shape[1]]).to(device)

    with torch.no_grad():
        encoded = model.encode_batch(waveform, wav_lengths)

    pred_ids = encoded.argmax(dim=-1)  # (batch, seq_len)

    transcription = []
    blank_id = 0

    for ids in pred_ids:
        # Décodage CTC : supprimer blanks et répétitions
        new_ids = []
        previous = None
        for i in ids.tolist():
            if i != blank_id and i != previous:
                new_ids.append(i)
            previous = i
        transcription.append(model.tokenizer.decode_ids(new_ids))


    end_time = time.perf_counter()
    duree = end_time - start_time

    return transcription, duree

def translate_one_batch_with_seamless(app, nom_model, batch: list[Chunk_audio]):
    if len(batch) == 0:
        logging.warning("translate_one_batch_with_seamless: batch is empty")
        return [], 0

    processor = app.state.models[nom_model]["processor"]
    model = app.state.models[nom_model]["model"]
    
    liste_audio_data = [chunk.get_audio_data() for chunk in batch]
    sampling_rate = batch[0].sr

    start_time = time.perf_counter()
    inputs = processor(audios=liste_audio_data, sampling_rate=sampling_rate, return_tensors="pt", attention_mask=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        predicted_ids = model.generate(**inputs, tgt_lang="fra")
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    end_time = time.perf_counter()

    duree = end_time - start_time

    return transcription, duree
