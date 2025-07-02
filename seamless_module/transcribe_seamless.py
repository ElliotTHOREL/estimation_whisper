#!/usr/bin/env python3
"""
Script de transcription audio -> texte avec SeamlessM4T (HuggingFace)
Usage :
    python transcribe_seamless.py chemin_audio.wav fra

- chemin_audio.wav : chemin vers le fichier audio à transcrire
- fra : code langue cible (ex : fra pour français, eng pour anglais, etc.)

Dépendances :
    pip install transformers datasets torch

Modèle utilisé : facebook/hf-seamless-m4t-medium
"""
import sys
from transformers import AutoProcessor, SeamlessM4TModel
import torch
import soundfile as sf

if len(sys.argv) < 3:
    print("Usage : python transcribe_seamless.py chemin_audio.wav fra")
    sys.exit(1)

audio_path = sys.argv[1]
tgt_lang = sys.argv[2]

# Charger l'audio
try:
    audio_array, sample_rate = sf.read(audio_path)
except Exception as e:
    print(f"Erreur lors de la lecture de l'audio : {e}")
    sys.exit(1)

# Charger le modèle et le processor
processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")

# Préparer l'entrée
inputs = processor(audios=audio_array, return_tensors="pt", sampling_rate=sample_rate)

# Générer la transcription texte
with torch.no_grad():
    output_tokens = model.generate(**inputs, tgt_lang=tgt_lang, generate_speech=False)
    transcription = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)

print(f"\nTranscription ({tgt_lang}) :\n{transcription}\n") 