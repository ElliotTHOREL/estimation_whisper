#!/usr/bin/env python3
"""
Gestionnaire spécialisé pour les modèles Whisper - Français uniquement
"""

from .base_model import BaseSpeechModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import torch
import numpy as np
from typing import Dict, List, Any
import logging
import os

logger = logging.getLogger(__name__)


class WhisperManager(BaseSpeechModel):
    """
    Gestionnaire pour les modèles Whisper d'OpenAI - Français uniquement
    """
    
    # Modèles Whisper disponibles
    AVAILABLE_MODELS = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base", 
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large": "openai/whisper-large-v2",
        "large-v3": "openai/whisper-large-v3"
    }
    
    def __init__(self, model_size: str = "small", device: str = "auto"):
        """
        Initialise le gestionnaire Whisper
        
        Args:
            model_size: Taille du modèle ("tiny", "base", "small", "medium", "large", "large-v3")
            device: Device à utiliser
        """
        model_name = f"openai/whisper-{model_size}"
        super().__init__(model_name, device)
        self.model_size = model_size
        self.use_pipeline = True  # Utilise le pipeline par défaut
        self.language = "fr"  # Toujours français dans ce projet
        self.processor = None
        self.model = None
        logger.info(f"Whisper {model_size} initialisé pour le français (langue: {self.language})")
    
    def _get_supported_languages(self) -> List[str]:
        """Whisper français uniquement dans notre cas"""
        return ["fr"]
    
    def _get_supported_formats(self) -> List[str]:
        """Formats audio supportés par Whisper"""
        return [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".mp4"]
    
    def load_model(self) -> None:
        if self.is_loaded:
            return
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=-1  # Force CPU pour éviter les erreurs de mémoire CUDA
        )
        self.is_loaded = True

    def _transcribe_audio_internal(self, audio: np.ndarray, language: str = None, **kwargs) -> Dict[str, Any]:
        """
        Transcription interne avec Whisper
        
        Args:
            audio: Audio préparé (16kHz)
            language: Langue (forcée à "fr" depuis .env)
            **kwargs: Arguments (non utilisés pour Whisper)
            
        Returns:
            Résultat de transcription
        """
        import torch
        # Préparation des inputs
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language or self.language, task="transcribe")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                forced_decoder_ids=forced_decoder_ids
            )
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"text": transcription.strip()}
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Retourne les modèles Whisper disponibles"""
        return cls.AVAILABLE_MODELS.copy()
    
    def get_model_info(self) -> Dict:
        """Informations étendues pour Whisper"""
        base_info = super().get_model_info()
        base_info.update({
            "model_size": self.model_size,
            "use_pipeline": self.use_pipeline,
            "available_sizes": list(self.AVAILABLE_MODELS.keys()),
            "framework": "whisper",
            "optimized_for": "français"
        })
        return base_info 