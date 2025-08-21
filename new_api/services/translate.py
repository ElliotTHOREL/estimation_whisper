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
    
    if len(batch) == 0:
        logging.warning("translate_one_batch: batch is empty")
        return [], 0

    type_modele = find_type_modele(nom_model)

    start_time = time.perf_counter()
    if type_modele in ("whisper", "wav2vec2", "kyutai", "seamless"): #PIPELINE GENERIQUE
        transcription= _translate_one_batch_generique(app, nom_model, batch, type_modele)
    elif type_modele in ("speechbrain_seq2seq", "speechbrain_ctc"): #PIPELINE SPEECHBRAIN (haut niveau)
        transcription= _translate_one_batch_speechbrain(app, nom_model, batch, type_modele)
    else:
        raise ValueError(f"Le modèle {nom_model} n'est pas supporté")

    end_time = time.perf_counter()
    duree = end_time - start_time

    return transcription, duree



def _translate_one_batch_generique(app, nom_model, batch, type_modele):
    #RECUPERATION DES DONNEES AUDIO
    liste_audio_data = [chunk.get_audio_data() for chunk in batch]
    sampling_rate = batch[0].sr

    #RECUPERATION DU MODELE
    processor = app.state.models[nom_model]["processor"]
    model = app.state.models[nom_model]["model"]

    #PROCESSING (audio -> features exploitables)
    if type_modele in ("seamless"): #particularité du processor seamless
        inputs = processor(audios=liste_audio_data, sampling_rate=sampling_rate, return_tensors="pt",padding=True)
    else:
        inputs = processor(liste_audio_data, sampling_rate=sampling_rate, return_tensors="pt",padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    #TOKENISATION (features -> tokens audio)
    if type_modele in ("wav2vec2"): #MODELE CTC, on utilise le logit 
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    else : #MODELE SEQ2SEQ (whisper, kyutai, seamless)
        generate_kwargs = {}
        if type_modele == "whisper": #spécialisation de la langue pour whisper
            generate_kwargs["language"] = "fr"
        elif type_modele == "seamless": #spécialisation de la langue pour seamless
            generate_kwargs["tgt_lang"] = "fra"
        # pour "kyutai", on laisse generate_kwargs vide (pas de langue spécialisée)

        with torch.no_grad():
            predicted_ids = model.generate(**inputs, **generate_kwargs)

    #DECODAGE (tokens audio -> texte)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription

def _translate_one_batch_speechbrain(app, nom_model, batch, type_modele):
    #RECUPERATION DES DONNEES AUDIO
    liste_audio_data = [chunk.get_audio_data() for chunk in batch]

    #RECUPERATION DU MODELE
    model = app.state.models[nom_model]["model"]

    #TENSORIFICATION
    audio_array = np.array(liste_audio_data)
    waveform = torch.tensor(audio_array, dtype=torch.float32).to(device)
    wav_lengths = torch.tensor([len(a) for a in liste_audio_data], dtype=torch.int32).to(device)

    if type_modele == "speechbrain_seq2seq": #EncoderDecoder ASR 
        #(PROCESSING + TOKENISATION + DECODAGE)
        #(audio -> texte)
        with torch.no_grad():
            transcription = model.transcribe_batch(waveform, wav_lengths) 
        transcription = [transcription[0][i] for i in range(len(transcription[0]))] #On ne garde que le texteet pas les ids_tokens

    elif type_modele == "speechbrain_ctc": #Encoder ASR
        #(PROCESSING + TOKENISATION)
        #(audio -> tokens audio)
        with torch.no_grad():
            encoded = model.encode_batch(waveform, wav_lengths)
        pred_ids = encoded.argmax(dim=-1)  # (batch, seq_len)

        #DECODAGE (à la main)
        # (tokens audio -> texte)
        transcription = []
        blank_id = 0

        # Décodage CTC : supprimer blanks et répétitions
        for ids in pred_ids.tolist():

            new_ids = []
            previous = None
            for i in ids:
                if i != blank_id and i != previous:
                    new_ids.append(i)
                previous = i

        transcription.append(model.tokenizer.decode_ids(new_ids))

    return transcription