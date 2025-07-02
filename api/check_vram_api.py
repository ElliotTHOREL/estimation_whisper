#!/usr/bin/env python3
"""
Script pour mesurer la consommation VRAM des modèles Whisper via l'API

Ce script :
1. Vérifie que l'API est accessible
2. Utilise les modèles déjà chargés dans l'API
3. Mesure la consommation VRAM réelle
4. Génère des recommandations

Usage:
    python api/check_vram_api.py
"""

import requests
import time
import psutil
import os
import torch
from typing import Dict, List, Optional
import json


class VRAMAnalyzer:
    """Analyseur VRAM utilisant l'API Whisper"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.test_audio = "cv-corpus-21.0-delta-2025-03-14/fr/clips/common_voice_fr_41911225.mp3"
        
    def check_api_health(self) -> bool:
        """Vérifie que l'API est accessible"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_gpu_memory(self) -> Dict:
        """Retourne l'usage mémoire GPU actuel"""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "cached": torch.cuda.memory_reserved() / 1024**3,      # GB
                "total": torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            }
        return {"allocated": 0, "cached": 0, "total": 0}
    
    def get_cache_info(self) -> Dict:
        """Récupère les informations du cache de l'API"""
        try:
            response = requests.get(f"{self.api_url}/cache/info")
            return response.json()
        except:
            return {}
    
    def measure_baseline(self) -> Dict:
        """Mesure la VRAM de base (avant chargement des modèles)"""
        # Vider le cache d'abord
        try:
            requests.post(f"{self.api_url}/cache/clear")
            time.sleep(2)
        except:
            pass
        
        return self.get_gpu_memory()
    
    def load_model_via_api(self, model_size: str) -> Dict:
        """Charge un modèle via l'API et mesure la VRAM"""
        print(f"🔍 Chargement du modèle {model_size} via API...")
        
        # Mesure avant
        gpu_before = self.get_gpu_memory()
        
        # Appel API pour charger le modèle
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.api_url}/transcribe_file",
                data={
                    "file_path": self.test_audio,
                    "model_size": model_size,
                    "language": "fr"
                },
                timeout=120
            )
            load_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                inference_time = result.get("inference_time", 0)
            else:
                print(f"❌ Erreur API: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"❌ Erreur lors du chargement de {model_size}: {e}")
            return {}
        
        # Mesure après
        time.sleep(1)  # Laisser le temps au GPU de se stabiliser
        gpu_after = self.get_gpu_memory()
        
        # Calcul des différences
        vram_used = gpu_after["allocated"] - gpu_before["allocated"]
        
        result = {
            "model": model_size,
            "vram_used_gb": vram_used,
            "vram_percentage": (vram_used / gpu_after["total"]) * 100 if gpu_after["total"] > 0 else 0,
            "load_time": load_time,
            "inference_time": inference_time,
            "gpu_total": gpu_after["total"]
        }
        
        print(f"  ⏱️  Temps de chargement: {load_time:.1f}s")
        print(f"  🚀 Temps d'inférence: {inference_time:.2f}s")
        print(f"  🎮 VRAM utilisée: {vram_used:.2f} GB ({result['vram_percentage']:.1f}%)")
        
        return result
    
    def analyze_current_state(self) -> Dict:
        """Analyse l'état actuel du cache et de la VRAM"""
        cache_info = self.get_cache_info()
        gpu_info = self.get_gpu_memory()
        
        return {
            "cache_info": cache_info,
            "gpu_memory": gpu_info,
            "cached_models_count": cache_info.get("cache_size", 0),
            "max_cache_size": cache_info.get("max_size", 0),
            "cached_models": cache_info.get("cached_models", [])
        }
    
    def estimate_vram_per_model(self, current_state: Dict) -> Dict:
        """Estime la VRAM par modèle basée sur l'état actuel"""
        models = current_state["cached_models"]
        total_vram = current_state["gpu_memory"]["allocated"]
        model_count = len(models)
        
        if model_count == 0:
            return {}
        
        # Estimation basée sur les tailles connues des modèles Whisper
        model_sizes = {
            "tiny": 0.07,
            "base": 0.14,
            "small": 0.47,
            "medium": 1.5,
            "large": 3.0,
            "large-v3": 3.0
        }
        
        estimates = {}
        total_estimated = 0
        
        for model_key in models:
            # Extraire le nom du modèle (format: "tiny_fr" -> "tiny")
            model_name = model_key.split("_")[0]
            if model_name in model_sizes:
                estimates[model_key] = model_sizes[model_name]
                total_estimated += model_sizes[model_name]
        
        return {
            "estimates": estimates,
            "total_estimated": total_estimated,
            "actual_vram": total_vram,
            "estimation_accuracy": abs(total_estimated - total_vram) / total_vram if total_vram > 0 else 0
        }
    
    def generate_recommendations(self, current_state: Dict, estimations: Dict):
        """Génère des recommandations basées sur l'analyse"""
        print(f"\n💡 RECOMMANDATIONS")
        print("=" * 50)
        
        gpu_total = current_state["gpu_memory"]["total"]
        gpu_used = current_state["gpu_memory"]["allocated"]
        cache_size = current_state["cached_models_count"]
        max_cache = current_state["max_cache_size"]
        
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"🎮 VRAM totale: {gpu_total:.1f} GB")
        print(f"🎮 VRAM utilisée: {gpu_used:.2f} GB ({(gpu_used/gpu_total)*100:.1f}%)")
        print(f"🗄️ Cache: {cache_size}/{max_cache} modèles")
        
        if estimations:
            print(f"\n📊 Estimation VRAM par modèle:")
            for model, vram in estimations["estimates"].items():
                print(f"  • {model}: ~{vram:.2f} GB")
            
            print(f"\n🎯 Total estimé: {estimations['total_estimated']:.2f} GB")
            print(f"🎯 VRAM réelle: {estimations['actual_vram']:.2f} GB")
            
        # Recommandations de capacité
        usage_percent = (gpu_used / gpu_total) * 100
        
        print(f"\n🚦 État de la VRAM:")
        if usage_percent < 30:
            print(f"  ✅ Excellent ({usage_percent:.1f}%) - Vous pouvez charger plus de modèles")
        elif usage_percent < 60:
            print(f"  ⚠️  Bon ({usage_percent:.1f}%) - Usage normal")
        elif usage_percent < 80:
            print(f"  ⚠️  Attention ({usage_percent:.1f}%) - Surveiller l'usage")
        else:
            print(f"  ❌ Critique ({usage_percent:.1f}%) - Risque de surcharge")
        
        # Recommandations de cache
        remaining_slots = max_cache - cache_size
        print(f"\n🗄️ Recommandations cache:")
        print(f"  • Places libres: {remaining_slots}/{max_cache}")
        
        if remaining_slots > 0:
            models_to_add = ["large-v3"] if "large-v3_fr" not in current_state["cached_models"] else []
            if models_to_add:
                print(f"  • Modèles suggérés: {', '.join(models_to_add)}")


def main():
    print("🔬 Analyse VRAM des modèles Whisper via API")
    print("=" * 50)
    
    analyzer = VRAMAnalyzer()
    
    # Vérification de l'API
    if not analyzer.check_api_health():
        print("❌ L'API Whisper n'est pas accessible sur http://localhost:8000")
        print("💡 Assurez-vous que l'API est démarrée avec: python api/start_api.py")
        return
    
    print("✅ API Whisper accessible")
    
    # Analyse de l'état actuel
    print(f"\n📊 Analyse de l'état actuel du cache...")
    current_state = analyzer.analyze_current_state()
    
    if not current_state["cached_models"]:
        print("⚠️  Aucun modèle en cache. Chargement des modèles de base...")
        
        # Charger quelques modèles pour analyse
        models_to_test = ["tiny", "base", "small"]
        for model in models_to_test:
            analyzer.load_model_via_api(model)
        
        # Re-analyser après chargement
        current_state = analyzer.analyze_current_state()
    
    # Estimation VRAM par modèle
    estimations = analyzer.estimate_vram_per_model(current_state)
    
    # Génération des recommandations
    analyzer.generate_recommendations(current_state, estimations)
    
    print(f"\n✅ Analyse terminée à {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main() 