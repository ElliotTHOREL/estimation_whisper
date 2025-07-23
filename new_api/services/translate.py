from services.audio_manager import  Audio_file, Chunk_audio

import torch
import time


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def translate_one (app, nom_model, id_audio, batch_audio):
    audio_file = Audio_file(id_audio, batch_audio)
    audio_file.get_liste_chunks()
    transcriptions = []
    total_duration = 0.0
    for chunk in audio_file.liste_chunks:
        print("translate_one", chunk.duration_ms, time.time())
        transcription, duree = translate_one_chunk(app, nom_model, chunk)
        transcriptions.append(transcription)
        total_duration += duree
    full_transcription = " ".join(transcriptions)
    return full_transcription, total_duration






def translate_one_chunk(app, nom_model, chunk):  
    if nom_model not in app.state.models:
        raise ValueError(f"Le modèle {nom_model} n'est pas chargé")

    if nom_model in ["w-tiny", "w-base", "w-small", "w-medium", "w-large-v2", "w-large-v3","b-w-large-v3","b-w-large-v3-distil","b-w-small-cv11","seamless-m4t-v2"]:
        transcription, duree = translate_one_chunk_with_whisper(app, nom_model, chunk)
    elif nom_model in ["w2-b-960h", "w2-large", "b-w2", "b-w2-1b"]:
        transcription, duree = translate_one_chunk_with_wav2vec(app, nom_model,  chunk)
    elif nom_model in ["kyutai-1b"]:
        transcription, duree = translate_one_chunk_with_kyutai(app, nom_model,  chunk)
    elif nom_model in ["sb-crdnn-fr", "sb-wav2vec2-fr"]:
        transcription, duree = translate_one_chunk_with_speechbrain(app, nom_model,  chunk)
    
    return transcription, duree

def translate_one_chunk_with_whisper(app, nom_model, chunk:Chunk_audio):
    processor = app.state.models[nom_model]["processor"]
    model = app.state.models[nom_model]["model"]

    audio_data, sampling_rate = chunk.get_audio_data(), chunk.sr

    start_time = time.perf_counter()
    inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", attention_mask=True)
    #inputs = {k: v.to(device) for k, v in inputs.items()}

    generate_kwargs = {}
    generate_kwargs["language"] = "fr"
    
    predicted_ids = model.generate(**inputs, **generate_kwargs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    end_time = time.perf_counter()

    duree = end_time - start_time

    return transcription[0], duree

def translate_one_chunk_with_wav2vec(app, nom_model, chunk:Chunk_audio):
    processor = app.state.models[nom_model]["processor"]
    model = app.state.models[nom_model]["model"]

    audio_data, sampling_rate = chunk.get_audio_data(), chunk.sr

    start_time = time.perf_counter()
    inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", attention_mask=True)
    #inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    end_time = time.perf_counter()

    duree = end_time - start_time

    return transcription, duree



def translate_one_chunk_with_kyutai(app, nom_model, chunk:Chunk_audio):
    
    processor = app.state.models[nom_model]["processor"]
    model = app.state.models[nom_model]["model"]
    
    audio_data, sampling_rate = chunk.get_audio_data(), chunk.sr

    start_time = time.perf_counter()
    inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", attention_mask=True)
    #inputs = {k: v.to(device) for k, v in inputs.items()}

    predicted_ids = model.generate(**inputs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    end_time = time.perf_counter()

    duree = end_time - start_time

    return transcription[0], duree



def translate_one_chunk_with_speechbrain(app, nom_model, chunk:Chunk_audio):
    audio_data, sampling_rate = chunk.get_audio_data(), chunk.sr

    model = app.state.models[nom_model]["model"]

    start_time = time.perf_counter()

    audio_np = chunk.get_audio_data()
    waveform = torch.tensor(audio_np).unsqueeze(0)
    waveform = waveform.to(model.device)
    transcription = model.transcribe_batch(waveform, torch.tensor([waveform.shape[1]]))

    end_time = time.perf_counter()
    duree = end_time - start_time
    
    return transcription, duree