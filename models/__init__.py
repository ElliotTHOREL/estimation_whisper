"""
Package pour les gestionnaires de modèles Speech-to-Text
Supporte Whisper, Wav2Vec2, SeamlessM4T et Gemma 3n
"""

from .base_model import BaseSpeechModel
from .whisper_manager import WhisperManager
from .wav2vec2_manager import Wav2Vec2Manager
from .seamless_manager import SeamlessManager
from .gemma3n_manager import Gemma3nManager
from .model_factory import ModelFactory

__version__ = "1.1.0"
__all__ = [
    "BaseSpeechModel",
    "WhisperManager", 
    "Wav2Vec2Manager",
    "SeamlessManager",
    "Gemma3nManager",
    "ModelFactory"
] 