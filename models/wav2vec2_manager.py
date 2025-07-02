#!/usr/bin/env python3
"""
Gestionnaire spécialisé pour les modèles Wav2Vec2 (Facebook) - Français uniquement
"""

from .base_model import BaseSpeechModel
from transformers import pipeline
import torch
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import os
from pathlib import Path
import librosa
from transformers.pipelines import pipeline

logger = logging.getLogger(__name__)


class Wav2Vec2Manager(BaseSpeechModel):
    """
    Gestionnaire pour les modèles Wav2Vec2 optimisés pour le français
    """
    
    # Modèles Wav2Vec2 français disponibles
    AVAILABLE_MODELS = {
        "xlsr-53-french": "facebook/wav2vec2-large-xlsr-53-french",
        "bhuang-french": "bhuang/asr-wav2vec2-french",
        "jonatasgrosman-xlsr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
        "jonatasgrosman-voxpopuli": "jonatasgrosman/wav2vec2-large-fr-voxpopuli-french",
        "speechbrain-cv14": "speechbrain/asr-wav2vec2-commonvoice-14-fr",
        "wasertech-cv9": "wasertech/wav2vec2-cv-fr-9"
    }
    
    def __init__(self, model_size: str = "xlsr-53-french", device: str = "auto"):
        """
        Initialise le gestionnaire Wav2Vec2
        
        Args:
            model_size: Taille du modèle ("xlsr-53-french", "base-french", "large-french")
            device: Device à utiliser
        """
        logger.info(f"DEBUG: Initialisation Wav2Vec2 - model_size={model_size}")
        logger.info(f"DEBUG: AVAILABLE_MODELS={self.AVAILABLE_MODELS}")
        
        if model_size not in self.AVAILABLE_MODELS:
            logger.warning(f"Model size '{model_size}' non trouvé, utilisation de 'xlsr-53-french' par défaut")
            model_size = "xlsr-53-french"
        
        model_name = self.AVAILABLE_MODELS[model_size]
        logger.info(f"DEBUG: model_name sélectionné={model_name}")
        
        if model_name is None:
            logger.error(f"DEBUG: ERREUR - model_name est None pour model_size={model_size}")
            raise ValueError(f"Model name is None for model_size={model_size}")
        
        super().__init__(model_name, device)
        self.model_size = model_size
        self.language = "fr"
        self.processor = None
        self.model = None
        
        logger.info(f"Wav2Vec2 {model_size} initialisé pour le français")
    
    def _get_supported_languages(self) -> List[str]:
        """Wav2Vec2 français uniquement"""
        return ["fr"]
    
    def _get_supported_formats(self) -> List[str]:
        """Formats audio supportés par Wav2Vec2"""
        return [".wav", ".flac", ".mp3", ".m4a"]
    
    def load_model(self) -> None:
        """Charge le modèle Wav2Vec2"""
        if self.is_loaded:
            logger.info(f"Modèle Wav2Vec2 {self.model_size} déjà chargé")
            return
        
        logger.info(f"DEBUG: load_model - model_name={self.model_name}")
        logger.info(f"DEBUG: load_model - device={self.device}")
        logger.info(f"Chargement du modèle Wav2Vec2 {self.model_size} ({self.model_name}) sur {self.device}")
        
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            logger.info(f"DEBUG: Import transformers réussi")
            
            logger.info(f"DEBUG: Chargement processor depuis {self.model_name}")
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            logger.info(f"DEBUG: Processor chargé")
            
            logger.info(f"DEBUG: Chargement modèle depuis {self.model_name}")
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
            logger.info(f"DEBUG: Modèle chargé")
            
            logger.info(f"DEBUG: Création pipeline avec model_name={self.model_name}")
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=-1  # Force CPU pour éviter les erreurs de mémoire CUDA
            )
            logger.info(f"DEBUG: Pipeline créé")
            
            self.is_loaded = True
            logger.info("Modèle Wav2Vec2 chargé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle Wav2Vec2: {e}")
            logger.error(f"DEBUG: Exception complète: {type(e).__name__}: {str(e)}")
            raise
    
    def _transcribe_audio_internal(self, audio: np.ndarray, language: str = None, **kwargs) -> Dict[str, Any]:
        """
        Transcrit un array numpy audio (méthode interne)
        
        Args:
            audio: Array numpy audio (16kHz, mono)
            language: Langue (ignorée pour Wav2Vec2)
            **kwargs: Options supplémentaires
            
        Returns:
            Dictionnaire avec la transcription
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Utiliser le pipeline pour la transcription
            result = self.pipeline(audio)
            
            return {
                "text": result.get("text", ""),
                "confidence": result.get("confidence", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la transcription Wav2Vec2: {e}")
            raise
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Retourne les modèles Wav2Vec2 français disponibles"""
        return cls.AVAILABLE_MODELS.copy()
    
    def get_model_info(self) -> Dict:
        """Informations étendues pour Wav2Vec2"""
        base_info = super().get_model_info()
        base_info.update({
            "model_size": self.model_size,
            "framework": "wav2vec2",
            "training_data": "French Common Voice + VoxPopuli",
            "available_sizes": list(self.AVAILABLE_MODELS.keys()),
            "optimized_for": "français"
        })
        return base_info 