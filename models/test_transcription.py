import torch
import librosa
from models.whisper_manager import WhisperManager
from models.wav2vec2_manager import Wav2Vec2Manager

AUDIO_PATH = "/data/elliot/estimattion_Whisper/evaluation/cv-corpus-21.0-delta-2025-03-14/fr/clips/common_voice_fr_41911225.mp3"
MODEL_TYPE = "wav2vec2"
MODEL_SIZE = "xlsr-53-french"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

audio, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True)
print(f"Audio chargé : {AUDIO_PATH} (shape={audio.shape}, sr={sr})")

if MODEL_TYPE == "whisper":
    manager = WhisperManager(model_size=MODEL_SIZE, device=DEVICE)
elif MODEL_TYPE == "wav2vec2":
    manager = Wav2Vec2Manager(model_size=MODEL_SIZE, device=DEVICE)
else:
    raise ValueError("Type de modèle inconnu.")

print(f"Chargement du modèle {MODEL_TYPE}:{MODEL_SIZE} sur {DEVICE} ...")
manager.load_model()

print("Transcription en cours ...")
result = manager.transcribe(audio)
print("\n--- Résultat ---")
print(result["text"])
print("\nInfos:", {k: v for k, v in result.items() if k != "text"}) 