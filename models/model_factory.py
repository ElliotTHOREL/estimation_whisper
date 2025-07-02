#!/usr/bin/env python3
"""
Factory pour créer et gérer les différents types de modèles Speech-to-Text
"""

from typing import Dict, List, Type, Union, Optional
from .base_model import BaseSpeechModel
from .whisper_manager import WhisperManager
from .wav2vec2_manager import Wav2Vec2Manager
from .seamless_manager import SeamlessManager
from .gemma3n_manager import Gemma3nManager
import logging
import os
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger(__name__)

# Charger le .env à la racine du projet
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
DEFAULT_LANGUAGE = os.environ.get("LANGUE", "fr")

class ModelFactory:
    """
    Factory pour créer et gérer les modèles Speech-to-Text
    """
    
    # Registry des types de modèles disponibles
    MODEL_TYPES = {
        "whisper": WhisperManager,
        "wav2vec2": Wav2Vec2Manager,
        "seamless": SeamlessManager,
        "gemma3n": Gemma3nManager
    }
    
    @classmethod
    def create_model(cls, 
                    model_type: str, 
                    model_size: str, 
                    device: str = "auto",
                    language: Optional[str] = None) -> BaseSpeechModel:
        """
        Crée une instance de modèle selon le type demandé
        
        Args:
            model_type: Type de modèle ("whisper", "wav2vec2", "seamless", "gemma3n")
            model_size: Taille/variant du modèle
            device: Device à utiliser
            language: Langue (forcée à "fr")
            
        Returns:
            Instance du gestionnaire de modèle
        """
        logger.info(f"DEBUG: ModelFactory.create_model - model_type={model_type}, model_size={model_size}")
        
        if model_type not in cls.MODEL_TYPES:
            raise ValueError(f"Type de modèle '{model_type}' non supporté. Types disponibles: {list(cls.MODEL_TYPES.keys())}")
        
        # Force la langue du .env si non spécifiée
        if language is None:
            language = DEFAULT_LANGUAGE
        
        manager_class = cls.MODEL_TYPES[model_type]
        logger.info(f"DEBUG: ModelFactory - manager_class={manager_class}")
        
        try:
            logger.info(f"DEBUG: ModelFactory - Création instance avec model_size={model_size}, device={device}")
            instance = manager_class(
                model_size=model_size,
                device=device
            )
            logger.info(f"DEBUG: ModelFactory - Instance créée avec succès: {instance}")
            logger.info(f"DEBUG: ModelFactory - model_name de l'instance: {getattr(instance, 'model_name', 'N/A')}")
            logger.info(f"Modèle {model_type}:{model_size} créé avec succès (langue: {language})")
            return instance
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du modèle {model_type}:{model_size}: {e}")
            raise
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, str]]:
        """
        Retourne tous les modèles disponibles par type
        
        Returns:
            Dictionnaire structuré des modèles disponibles
        """
        available = {}
        
        for model_type, manager_class in cls.MODEL_TYPES.items():
            try:
                available[model_type] = manager_class.get_available_models()
            except Exception as e:
                logger.warning(f"Erreur lors de la récupération des modèles {model_type}: {e}")
                available[model_type] = {}
        
        return available
    
    @classmethod
    def get_model_info(cls, model_type: str, model_size: str) -> Dict:
        """
        Obtient les informations sur un modèle spécifique sans le charger
        
        Args:
            model_type: Type de modèle
            model_size: Taille du modèle
            
        Returns:
            Informations sur le modèle
        """
        if model_type not in cls.MODEL_TYPES:
            raise ValueError(f"Type de modèle '{model_type}' non supporté")
        
        manager_class = cls.MODEL_TYPES[model_type]
        available_models = manager_class.get_available_models()
        
        if model_size not in available_models:
            raise ValueError(f"Modèle {model_type}:{model_size} non disponible. Modèles {model_type} disponibles: {list(available_models.keys())}")
        
        return {
            "model_type": model_type,
            "model_size": model_size,
            "model_name": available_models[model_size],
            "supported_language": "fr",
            "manager_class": manager_class.__name__
        }
    
    @classmethod
    def list_all_models(cls) -> List[Dict]:
        """
        Liste tous les modèles disponibles dans un format unifié
        
        Returns:
            Liste de tous les modèles avec leurs métadonnées
        """
        all_models = []
        available = cls.get_available_models()
        
        for model_type, models in available.items():
            for model_size, model_name in models.items():
                all_models.append({
                    "id": f"{model_type}:{model_size}",
                    "model_type": model_type,
                    "model_size": model_size,
                    "model_name": model_name,
                    "language": "fr",
                    "framework": model_type
                })
        
        return all_models
    
    @classmethod
    def validate_model(cls, model_type: str, model_size: str) -> bool:
        """
        Valide qu'un modèle existe et peut être créé
        
        Args:
            model_type: Type de modèle
            model_size: Taille du modèle
            
        Returns:
            True si le modèle est valide, False sinon
        """
        try:
            cls.get_model_info(model_type, model_size)
            return True
        except (ValueError, KeyError):
            return False
    
    @classmethod
    def get_recommended_models(cls) -> Dict[str, str]:
        """
        Retourne les modèles recommandés par type pour le français
        """
        return {
            "whisper": "base",
            "wav2vec2": "xlsr-53-french",
            "seamless": "large",
            "gemma3n": "e2b"
        }
    
    @classmethod
    def estimate_vram_usage(cls, model_type: str, model_size: str) -> Dict[str, Union[int, str]]:
        """
        Estime l'utilisation VRAM approximative d'un modèle
        
        Args:
            model_type: Type de modèle
            model_size: Taille du modèle
            
        Returns:
            Estimation de l'utilisation VRAM en MB
        """
        # Estimations approximatives en MB (peuvent varier selon le batch size, etc.)
        vram_estimates = {
            "whisper": {
                "tiny": {"vram_mb": 150, "description": "Très léger"},
                "base": {"vram_mb": 300, "description": "Léger"},
                "small": {"vram_mb": 600, "description": "Moyen"},
                "medium": {"vram_mb": 1200, "description": "Lourd"},
                "large": {"vram_mb": 2400, "description": "Très lourd"},
                "large-v3": {"vram_mb": 2400, "description": "Très lourd"}
            },
            "wav2vec2": {
                "base-french": {"vram_mb": 400, "description": "Léger"},
                "large-french": {"vram_mb": 800, "description": "Moyen"},
                "xlsr-53-french": {"vram_mb": 1000, "description": "Moyen-lourd"},
                "xlsr-300m-french": {"vram_mb": 600, "description": "Moyen"},
                "bhuang-french": {"vram_mb": 800, "description": "Moyen"},
                "jonatasgrosman-xlsr": {"vram_mb": 1000, "description": "Moyen-lourd"},
                "jonatasgrosman-voxpopuli": {"vram_mb": 1000, "description": "Moyen-lourd"},
                "speechbrain-cv14": {"vram_mb": 800, "description": "Moyen"},
                "wasertech-cv9": {"vram_mb": 600, "description": "Moyen"}
            },
            "seamless": {
                "large": {"vram_mb": 4000, "description": "SeamlessM4T v2 Large"},
                "medium": {"vram_mb": 2000, "description": "SeamlessM4T v1 Medium"}
            },
            "gemma3n": {
                "e2b": {"vram_mb": 3000, "description": "Gemma 3n E2B (2B effective, 5B total)"},
                "e4b": {"vram_mb": 4500, "description": "Gemma 3n E4B (4B effective, 8B total)"}
            }
        }
        
        if model_type in vram_estimates and model_size in vram_estimates[model_type]:
            return vram_estimates[model_type][model_size]
        else:
            return {"vram_mb": 1000, "description": "Estimation non disponible"}


# Fonctions utilitaires pour simplifier l'usage
def create_whisper_model(model_size: str = "base", device: str = "auto") -> BaseSpeechModel:
    """Raccourci pour créer un modèle Whisper"""
    return ModelFactory.create_model("whisper", model_size, device)

def create_wav2vec2_model(model_size: str = "xlsr-53-french", device: str = "auto") -> BaseSpeechModel:
    """Raccourci pour créer un modèle Wav2Vec2"""
    return ModelFactory.create_model("wav2vec2", model_size, device)

def create_gemma3n_model(model_size: str = "e2b", device: str = "auto") -> BaseSpeechModel:
    """Raccourci pour créer un modèle Gemma 3n"""
    return ModelFactory.create_model("gemma3n", model_size, device) 