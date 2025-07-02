#!/usr/bin/env python3
"""
Gestionnaire spécialisé pour les modèles Gemma 3n (Google DeepMind)
Modèles multimodaux optimisés pour les appareils avec capacités audio
"""

from .base_model import BaseSpeechModel
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import numpy as np
from typing import Dict, List, Any
import logging
import warnings

logger = logging.getLogger(__name__)

class Gemma3nManager(BaseSpeechModel):
    """
    Gestionnaire pour les modèles Gemma 3n multimodaux avec capacités audio
    Optimisés pour la transcription et traduction speech-to-text
    """
    
    # Modèles Gemma 3n disponibles sur HuggingFace
    AVAILABLE_MODELS = {
        "e2b": "google/gemma-3n-E2B-it",
        "e4b": "google/gemma-3n-E4B-it"
    }
    
    def __init__(self, model_size: str = "e2b", device: str = "auto"):
        """
        Initialise le gestionnaire Gemma 3n
        
        Args:
            model_size: Taille du modèle ("e2b", "e4b")
            device: Device à utiliser
        """
        if model_size not in self.AVAILABLE_MODELS:
            logger.warning(f"Model size '{model_size}' non trouvé, utilisation de 'e2b' par défaut")
            model_size = "e2b"
        
        model_name = self.AVAILABLE_MODELS[model_size]
        super().__init__(model_name, device)
        self.model_size = model_size
        self.language = "fr"  # Français par défaut
        self.processor = None
        self.model = None
        
        logger.info(f"Gemma 3n {model_size} initialisé pour le français (multilingual supporté)")
    
    def _get_supported_languages(self) -> List[str]:
        """Gemma 3n supporte plus de 140 langues, voici les principales"""
        return [
            "fr", "en", "es", "de", "ja", "ko", "zh", "pt", "it", "ru", 
            "ar", "hi", "th", "vi", "nl", "pl", "sv", "da", "no", "fi"
        ]
    
    def _get_supported_formats(self) -> List[str]:
        """Formats audio supportés par Gemma 3n"""
        return [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]
    
    def load_model(self) -> None:
        """Charge le modèle Gemma 3n"""
        if self.is_loaded:
            logger.info(f"Modèle Gemma 3n {self.model_size} déjà chargé")
            return
        
        logger.info(f"Chargement du modèle Gemma 3n {self.model_size} ({self.model_name}) sur {self.device}")
        
        try:
            # Suppression des warnings pour les modèles en preview
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                logger.info("Chargement du processor Gemma 3n...")
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                except Exception as e:
                    logger.warning(f"Processor AutoProcessor non disponible: {e}")
                    # Fallback: utiliser seulement le tokenizer pour l'instant
                    from transformers import AutoTokenizer
                    self.processor = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                    logger.info("Utilisation du tokenizer uniquement (mode text-only)")
                
                logger.info("Chargement du modèle Gemma 3n...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else "cpu",
                    trust_remote_code=True
                )
                
                if self.device == "cpu":
                    self.model = self.model.to("cpu")
            
            self.is_loaded = True
            logger.info("Modèle Gemma 3n chargé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle Gemma 3n: {e}")
            logger.info("Note: Gemma 3n est en preview, vérifiez les permissions HuggingFace")
            raise
    
    def _transcribe_audio_internal(self, audio: np.ndarray, language: str = None, **kwargs) -> Dict[str, Any]:
        """
        Transcrit un array numpy audio avec Gemma 3n
        
        Args:
            audio: Array numpy audio (16kHz, mono)
            language: Langue cible (optionnelle)
            **kwargs: Options supplémentaires
            
        Returns:
            Dictionnaire avec la transcription
        """
        if not self.is_loaded:
            self.load_model()
        
        # Vérifications de sécurité
        if self.model is None or self.processor is None:
            logger.error("Modèle ou processor non chargé correctement")
            used_language = str(language) if language is not None else str(self.language)
            raise RuntimeError(f"Modèle ou processor non chargé correctement pour Gemma 3n (langue demandée : {used_language})")
        
        target_language = language or self.language
        
        try:
            # Vérifier si on a les capacités audio complètes
            if hasattr(self.processor, '__class__') and hasattr(self.processor, 'feature_extractor'):
                # Mode multimodal complet
                logger.info("Mode multimodal avec audio")
                
                # Préparation du prompt pour la transcription audio
                if target_language == "fr":
                    prompt = "Transcris cet audio en français:"
                elif target_language == "en":
                    prompt = "Transcribe this audio to English:"
                else:
                    prompt = f"Transcribe this audio to {target_language}:"
                
                # Préparation des inputs pour Gemma 3n
                inputs = self.processor(
                    text=prompt,
                    audio=audio,
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                
                # LOG: Afficher les clés et shapes des inputs
                for k, v in inputs.items():
                    if hasattr(v, 'shape'):
                        logger.info(f"Input key: {k}, shape: {v.shape}")
                    else:
                        logger.info(f"Input key: {k}, type: {type(v)}")
                
                # Déplacement vers le device approprié
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                with torch.no_grad():
                    # Génération avec Gemma 3n
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                    
                    # Décodage de la réponse
                    generated_text = self.processor.decode(
                        outputs[0], 
                        skip_special_tokens=True
                    )
                    
                    # Extraction de la transcription (après le prompt)
                    transcription = generated_text.replace(prompt, "").strip()
                
                return {
                    "text": transcription,
                    "confidence": 0.95,
                    "model_size": self.model_size,
                    "framework": "gemma-3n"
                }
            else:
                # Mode text-only (fallback)
                logger.error("Mode text-only - capacités audio non disponibles pour Gemma 3n")
                raise RuntimeError("Capacités audio non disponibles pour Gemma 3n sur ce processor")
            
        except Exception as e:
            logger.error(f"Erreur lors de la transcription Gemma 3n: {e}")
            raise e
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Retourne les modèles Gemma 3n disponibles"""
        return cls.AVAILABLE_MODELS.copy()
    
    def get_model_info(self) -> Dict:
        """Informations étendues pour Gemma 3n"""
        base_info = super().get_model_info()
        base_info.update({
            "model_size": self.model_size,
            "framework": "gemma-3n",
            "multimodal": True,
            "audio_capabilities": [
                "speech_recognition",
                "speech_translation", 
                "audio_analysis"
            ],
            "effective_parameters": "2B" if self.model_size == "e2b" else "4B",
            "total_parameters": "5B" if self.model_size == "e2b" else "8B",
            "optimization": "mobile_first",
            "available_sizes": list(self.AVAILABLE_MODELS.keys()),
            "languages_supported": 140,
            "preview_status": False
        })
        return base_info
    
    def get_model_requirements(self) -> Dict:
        """Exigences spécifiques pour Gemma 3n"""
        return {
            "huggingface_login": True,
            "license_agreement": "Gemma License required",
            "preview_model": False,
            "min_transformers_version": "4.30.0",
            "recommended_torch_version": "2.0.0+",
            "trust_remote_code": True
        } 