#!/usr/bin/env python3
"""
API universelle pour la transcription Speech-to-Text
Supporte Whisper et Wav2Vec2
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import threading
import time
import os
import tempfile
import shutil
from typing import Dict, List, Optional
import logging
from contextlib import asynccontextmanager
from pathlib import Path

# Import pour charger le .env
try:
    from dotenv import load_dotenv
    # Charge le .env depuis la racine du projet
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logging.info(f"Fichier .env chargé depuis {env_path}")
except ImportError:
    logging.warning("python-dotenv non installé, variables d'environnement du .env non chargées")

# Import de nos gestionnaires de modèles
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models import ModelFactory, BaseSpeechModel

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration depuis les variables d'environnement
DEFAULT_LANGUAGE = os.environ.get("LANGUE", "fr")
MAX_MODELS_CACHE = int(os.environ.get("MAX_MODELS_CACHE", "20"))

logger.info(f"Configuration API: LANGUE={DEFAULT_LANGUAGE}, MAX_MODELS_CACHE={MAX_MODELS_CACHE}")


class UniversalModelCache:
    """Cache intelligent pour tous les types de modèles Speech-to-Text"""
    
    def __init__(self, max_models: int = MAX_MODELS_CACHE):
        self.cache: Dict[str, BaseSpeechModel] = {}
        self.access_times: Dict[str, float] = {}
        self.max_models = max_models
        self.lock = threading.RLock()
        
        logger.info(f"Cache initialisé avec capacité maximale: {max_models} modèles")
        
    def get_model(self, model_type: str, model_size: str, language: str = None) -> BaseSpeechModel:
        """Récupère un modèle du cache ou le charge"""
        # Utilise la langue par défaut si non spécifiée
        language = language or DEFAULT_LANGUAGE
        cache_key = f"{model_type}:{model_size}:{language}"
        
        with self.lock:
            if cache_key in self.cache:
                self.access_times[cache_key] = time.time()
                logger.info(f"Modèle {cache_key} récupéré du cache")
                return self.cache[cache_key]
            
            # Si on atteint la limite, on supprime le modèle le moins récemment utilisé
            if len(self.cache) >= self.max_models:
                self._evict_lru()
            
            # Chargement du nouveau modèle
            logger.info(f"Chargement du modèle {cache_key}")
            try:
                model_instance = ModelFactory.create_model(
                    model_type=model_type,
                    model_size=model_size,
                    language=language
                )
                model_instance.load_model()
                
                self.cache[cache_key] = model_instance
                self.access_times[cache_key] = time.time()
                
                logger.info(f"Modèle {cache_key} chargé et mis en cache")
                return model_instance
                
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle {cache_key}: {e}")
                raise HTTPException(status_code=500, detail=f"Impossible de charger le modèle {model_type}:{model_size}: {str(e)}")
    
    def _evict_lru(self):
        """Supprime le modèle le moins récemment utilisé"""
        if not self.cache:
            return
            
        lru_key = min(self.access_times, key=self.access_times.get)
        logger.info(f"Suppression du modèle {lru_key} du cache (LRU)")
        
        # Décharge le modèle proprement
        if lru_key in self.cache:
            self.cache[lru_key].unload_model()
            del self.cache[lru_key]
        
        if lru_key in self.access_times:
            del self.access_times[lru_key]
    
    def get_cache_info(self) -> Dict:
        """Retourne des informations sur le cache"""
        models_info = {}
        for key, model in self.cache.items():
            models_info[key] = {
                "model_info": model.get_model_info(),
                "last_access": time.ctime(self.access_times[key]),
                "vram_estimate": ModelFactory.estimate_vram_usage(
                    model.model_type, 
                    getattr(model, 'model_size', 'unknown')
                )
            }
        
        return {
            "cached_models": list(self.cache.keys()),
            "cache_size": len(self.cache),
            "max_size": self.max_models,
            "models_details": models_info,
            "default_language": DEFAULT_LANGUAGE
        }
    
    def clear_cache(self):
        """Vide le cache"""
        with self.lock:
            # Décharge tous les modèles proprement
            for model in self.cache.values():
                try:
                    model.unload_model()
                except Exception as e:
                    logger.warning(f"Erreur lors du déchargement d'un modèle: {e}")
            
            self.cache.clear()
            self.access_times.clear()
            logger.info("Cache vidé complètement")


# Instance globale du cache
model_cache = UniversalModelCache(max_models=MAX_MODELS_CACHE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie de l'application"""
    logger.info(f"Démarrage de l'API Speech universelle (langue: {DEFAULT_LANGUAGE})")
    yield
    logger.info("Arrêt de l'API Speech - Nettoyage du cache")
    model_cache.clear_cache()


# Création de l'application FastAPI
app = FastAPI(
    title="Universal Speech-to-Text API",
    description="API universelle pour la transcription audio avec Whisper et Wav2Vec2",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "API Speech-to-Text Universelle",
        "version": "2.0.0",
        "default_language": DEFAULT_LANGUAGE,
        "supported_model_types": list(ModelFactory.MODEL_TYPES.keys()),
        "available_endpoints": [
            "/transcribe",
            "/transcribe_file", 
            "/load_model",
            "/models",
            "/models/recommended",
            "/models/vram",
            "/cache/info",
            "/cache/clear",
            "/health"
        ]
    }


@app.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    model_type: str = Form("whisper"),
    model_size: str = Form("base"),
    language: str = Form(DEFAULT_LANGUAGE),
    return_timestamps: bool = Form(False)
):
    """
    Transcrit un fichier audio uploadé
    
    Args:
        audio_file: Fichier audio à transcrire
        model_type: Type de modèle ("whisper", "wav2vec2")
        model_size: Taille/variant du modèle
        language: Langue de transcription (défaut depuis .env)
        return_timestamps: Retourner les timestamps
    """
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="Aucun fichier fourni")
    
    # Validation du modèle
    if not ModelFactory.validate_model(model_type, model_size):
        available = ModelFactory.get_available_models()
        raise HTTPException(
            status_code=400, 
            detail=f"Modèle {model_type}:{model_size} non supporté. Modèles disponibles: {available}"
        )
    
    # Vérification du type de fichier
    allowed_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.mp4'}
    file_extension = os.path.splitext(audio_file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Type de fichier non supporté. Extensions autorisées: {allowed_extensions}"
        )
    
    try:
        # Sauvegarde temporaire du fichier
        audio_bytes = await audio_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Récupération du modèle
        model = model_cache.get_model(model_type, model_size, language)
        
        # Charger l'audio pour passer le language à transcribe
        import librosa
        audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
        result = model.transcribe(audio, language=language, return_timestamps=return_timestamps)
        
        os.remove(tmp_path)
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Erreur lors de la transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de transcription: {str(e)}")


@app.post("/transcribe_file")
async def transcribe_file_path(
    file_path: str = Form(...),
    model_type: str = Form("whisper"),
    model_size: str = Form("base"),
    language: str = Form(DEFAULT_LANGUAGE),
    return_timestamps: bool = Form(False)
):
    """
    Transcrit un fichier audio via son chemin sur le serveur
    """
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Fichier non trouvé: {file_path}")

    # Validation du modèle
    if not ModelFactory.validate_model(model_type, model_size):
        available = ModelFactory.get_available_models()
        raise HTTPException(
            status_code=400, 
            detail=f"Modèle {model_type}:{model_size} non supporté. Modèles disponibles: {available}"
        )

    try:
        # Récupération du modèle
        model = model_cache.get_model(model_type, model_size, language)
        
        # Charger l'audio pour passer le language à transcribe
        import librosa
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        result = model.transcribe(audio, language=language, return_timestamps=return_timestamps)
        
        # Ajout métadonnées API
        result.update({
            "api_version": "2.0.0",
            "file_path": file_path
        })
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Erreur lors de la transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de transcription: {str(e)}")


@app.get("/models")
async def get_available_models():
    """Retourne tous les modèles disponibles par type"""
    try:
        available = ModelFactory.get_available_models()
        all_models = ModelFactory.list_all_models()
        
        return {
            "models_by_type": available,
            "all_models": all_models,
            "total_models": len(all_models),
            "supported_types": list(ModelFactory.MODEL_TYPES.keys()),
            "default_language": DEFAULT_LANGUAGE
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des modèles: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/models/recommended")
async def get_recommended_models():
    """Retourne les modèles recommandés par type"""
    try:
        recommended = ModelFactory.get_recommended_models()
        
        # Ajout des estimations VRAM pour les modèles recommandés
        recommendations_with_vram = {}
        for model_type, model_size in recommended.items():
            vram_info = ModelFactory.estimate_vram_usage(model_type, model_size)
            recommendations_with_vram[model_type] = {
                "model_size": model_size,
                "full_name": ModelFactory.get_model_info(model_type, model_size)["model_name"],
                "vram_estimate_mb": vram_info["vram_mb"],
                "description": vram_info["description"]
            }
        
        return {
            "recommended_models": recommendations_with_vram,
            "language": DEFAULT_LANGUAGE,
            "note": "Modèles recommandés pour un usage général en français"
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des recommandations: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/models/vram")
async def get_vram_estimates():
    """Retourne les estimations VRAM pour tous les modèles"""
    try:
        available = ModelFactory.get_available_models()
        vram_estimates = {}
        
        for model_type, models in available.items():
            vram_estimates[model_type] = {}
            for model_size in models.keys():
                vram_info = ModelFactory.estimate_vram_usage(model_type, model_size)
                vram_estimates[model_type][model_size] = vram_info
        
        return {
            "vram_estimates": vram_estimates,
            "unit": "MB",
            "note": "Estimations approximatives selon la configuration et le batch size"
        }
    except Exception as e:
        logger.error(f"Erreur lors du calcul des estimations VRAM: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/cache/info")
async def get_cache_info():
    """Informations sur le cache des modèles"""
    try:
        return model_cache.get_cache_info()
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos cache: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.post("/cache/clear")
async def clear_cache():
    """Vide le cache des modèles"""
    try:
        model_cache.clear_cache()
        return {
            "message": "Cache vidé avec succès",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Erreur lors du vidage du cache: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/health")
async def health_check():
    """Vérification de santé de l'API"""
    return {
        "status": "ok",
        "version": "2.0.0",
        "default_language": DEFAULT_LANGUAGE,
        "max_cache_size": MAX_MODELS_CACHE,
        "current_cache_size": len(model_cache.cache),
        "supported_models": len(ModelFactory.list_all_models()),
        "timestamp": time.time()
    }


@app.post("/load_model")
async def load_model(
    model_type: str = Form("whisper"),
    model_size: str = Form("base"),
    language: str = Form(DEFAULT_LANGUAGE)
):
    """
    Charge explicitement un modèle en mémoire (sans transcrire).
    Utile pour précharger les modèles à l'avance.
    """
    # Validation du modèle
    if not ModelFactory.validate_model(model_type, model_size):
        available = ModelFactory.get_available_models()
        raise HTTPException(
            status_code=400,
            detail=f"Modèle {model_type}:{model_size} non supporté. Modèles disponibles: {available}"
        )
    try:
        model = model_cache.get_model(model_type, model_size, language)
        info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
        return {
            "message": f"Modèle {model_type}:{model_size} chargé en mémoire.",
            "model_type": model_type,
            "model_size": model_size,
            "language": language,
            "model_info": info
        }
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du modèle: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="API Speech-to-Text Universelle")
    parser.add_argument("--host", default="0.0.0.0", help="Host de l'API")
    parser.add_argument("--port", type=int, default=8000, help="Port de l'API")
    parser.add_argument("--reload", action="store_true", help="Mode reload pour le développement")
    args = parser.parse_args()
    
    logger.info(f"Démarrage API sur {args.host}:{args.port}")
    
    uvicorn.run(
        "speech_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    ) 