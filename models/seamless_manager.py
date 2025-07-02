#!/usr/bin/env python3
"""
Gestionnaire spécialisé pour les modèles SeamlessM4T (v1 et v2)
"""

from .base_model import BaseSpeechModel
from transformers import AutoProcessor, SeamlessM4Tv2Model, SeamlessM4TModel
import torch
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class SeamlessManager(BaseSpeechModel):
    """
    Gestionnaire pour les modèles SeamlessM4T v2 Large (facebook/seamless-m4t-v2-large) et v1 Medium (facebook/hf-seamless-m4t-medium)
    """
    AVAILABLE_MODELS = {
        "large": "facebook/seamless-m4t-v2-large",
        "medium": "facebook/hf-seamless-m4t-medium"
    }

    def __init__(self, model_size: str = "large", device: str = "auto"):
        model_name = self.AVAILABLE_MODELS.get(model_size, self.AVAILABLE_MODELS["large"])
        super().__init__(model_name, device)
        self.model_size = model_size
        self.language = "fr"
        self.processor = None
        self.model = None
        logger.info(f"SeamlessM4T {model_size} initialisé pour le français")

    def _get_supported_languages(self) -> List[str]:
        return ["fr"]

    def _get_supported_formats(self) -> List[str]:
        return [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

    def load_model(self) -> None:
        if self.is_loaded:
            logger.info(f"Modèle SeamlessM4T {self.model_size} déjà chargé")
            return
        
        # Configuration adaptée pour CPU
        model_kwargs = {}
        if self.device == "cpu":
            # Sur CPU, utiliser float32 et pas de flash attention
            model_kwargs = {
                "torch_dtype": torch.float32
            }
        else:
            # Configuration originale pour GPU (si on revient au GPU plus tard)
            if self.model_size == "large":
                model_kwargs = {
                    "attn_implementation": "flash_attention_2",
                    "torch_dtype": torch.float16
                }
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        if self.model_size == "large":
            self.model = SeamlessM4Tv2Model.from_pretrained(self.model_name, **model_kwargs).to(self.device)
        else:
            self.model = SeamlessM4TModel.from_pretrained(self.model_name, **model_kwargs).to(self.device)
        
        self.is_loaded = True
        logger.info(f"Modèle SeamlessM4T {self.model_size} chargé avec succès")

    def _transcribe_audio_internal(self, audio: np.ndarray, language: str = None, **kwargs) -> dict:
        sampling_rate = 16000
        lang = language or self.language
        
        # Conversion du code de langue pour SeamlessM4T
        if lang == "fr":
            lang = "fra"
        
        try:
            # Traitement de l'audio
            inputs = self.processor(audios=audio, return_tensors="pt", sampling_rate=sampling_rate)
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                # Génération avec paramètres corrects pour SeamlessM4T
                generation_kwargs = {
                    "tgt_lang": lang,
                    "generate_speech": False,
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": 256
                }
                
                # Suppression des paramètres qui causent des problèmes
                if "input_ids" in inputs:
                    del inputs["input_ids"]
                
                output_tokens = self.model.generate(**inputs, **generation_kwargs)
                
                # Décodage des tokens
                tokens = output_tokens.tolist() if hasattr(output_tokens, 'tolist') else output_tokens
                if isinstance(tokens[0], list):
                    transcription = self.processor.batch_decode(tokens, skip_special_tokens=True)[0]
                else:
                    transcription = self.processor.decode(tokens, skip_special_tokens=True)
                
            return {
                "text": transcription.strip(), 
                "model_size": self.model_size, 
                "framework": f"seamless-m4t-{self.model_size}"
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la transcription SeamlessM4T: {e}")
            raise e

    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        return cls.AVAILABLE_MODELS.copy()

    def get_model_info(self) -> Dict:
        base_info = super().get_model_info()
        base_info.update({
            "model_size": self.model_size,
            "framework": f"seamless-m4t-{self.model_size}",
            "available_sizes": list(self.AVAILABLE_MODELS.keys()),
            "optimized_for": "multilingue"
        })
        return base_info 