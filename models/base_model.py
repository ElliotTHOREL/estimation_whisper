#!/usr/bin/env python3
"""
Classe de base abstraite pour tous les gestionnaires de modèles Speech-to-Text
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional
import torch
import librosa
import numpy as np
import logging
import time
import os
from pathlib import Path

# Import pour charger le .env
try:
    from dotenv import load_dotenv
    # Charge le .env depuis la racine du projet
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # Si python-dotenv n'est pas installé, on continue sans
    pass

logger = logging.getLogger(__name__)


class BaseSpeechModel(ABC):
    """
    Classe abstraite de base pour tous les modèles Speech-to-Text
    """
    
    def __init__(self, model_name: str, device: str = "auto"):
        """
        Initialise le gestionnaire de modèle
        
        Args:
            model_name: Nom/identifiant du modèle
            device: Device à utiliser ("cpu", "cuda", "auto")
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        
        self.pipeline = None
        self.is_loaded = False
        
        # Métadonnées du modèle
        self.model_type = self.__class__.__name__.replace("Manager", "").lower()
        self.supported_languages = self._get_supported_languages()
        self.supported_formats = self._get_supported_formats()
        
        logger.info(f"Modèle initialisé")
    
    def _get_device(self, device: str) -> str:
        """Détermine le device optimal - forcé sur CPU pour éviter les erreurs de mémoire GPU"""
        if device == "auto":
            # Forcer CPU pour éviter les problèmes de mémoire CUDA
            return "cpu"
        return device
    
    @abstractmethod
    def _get_supported_languages(self) -> List[str]:
        """Retourne les langues supportées par ce modèle"""
        pass
    
    @abstractmethod
    def _get_supported_formats(self) -> List[str]:
        """Retourne les formats audio supportés"""
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """Charge le modèle en mémoire"""
        pass
    
    @abstractmethod
    def _transcribe_audio_internal(self, audio: np.ndarray, language: str = None, **kwargs) -> Dict:
        """Méthode abstraite pour la transcription interne d'un array audio"""
        pass
    
    def transcribe_file(self, audio_path: str) -> Dict:
        """
        Transcrit un fichier audio (méthode publique pour fichiers)
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            Dictionnaire standardisé avec la transcription
        """
        if not self.is_loaded:
            self.load_model()
        
        # Charger l'audio en array numpy
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Utiliser la méthode transcribe pour array
        return self.transcribe(audio)
    
    def transcribe(self, audio: np.ndarray, language: str = None, **kwargs) -> Dict:
        """
        Transcrit un array numpy audio (16kHz, mono) directement.
        Args:
            audio: np.ndarray (mono, 16kHz)
            language: langue cible (optionnelle)
            **kwargs: options spécifiques au modèle
        Returns:
            Dictionnaire standardisé avec la transcription
        """
        if not self.is_loaded:
            self.load_model()
        
        language = language or getattr(self, 'language', None)
        if language and language not in self.supported_languages:
            logger.warning(f"Langue '{language}' non officiellement supportée par {self.model_type}. Langues supportées: {self.supported_languages}")
        
        start_time = time.time()
        result = self._transcribe_audio_internal(audio, language=language, **kwargs)
        inference_time = time.time() - start_time
        
        standardized_result = {
            "text": result.get("text", ""),
            "model_type": self.model_type,
            "model_name": self.model_name,
            "language": language,
            "inference_time": inference_time,
            "success": True
        }
        
        # Ajouter les autres champs du résultat
        for key, value in result.items():
            if key not in standardized_result:
                standardized_result[key] = value
                
        return standardized_result
    
    def get_model_info(self) -> Dict:
        """Retourne les informations sur le modèle"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "supported_languages": self.supported_languages,
            "supported_formats": self.supported_formats
        }
    
    def unload_model(self) -> None:
        """Décharge le modèle de la mémoire"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        self.is_loaded = False
        # Libération mémoire GPU si disponible
        try:
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info(f"Modèle {self.model_name} déchargé de la mémoire")

    def __del__(self):
        """Destructeur - s'assure que le modèle est déchargé (robuste)"""
        try:
            if hasattr(self, 'is_loaded') and self.is_loaded:
                self.unload_model()
        except Exception:
            pass 