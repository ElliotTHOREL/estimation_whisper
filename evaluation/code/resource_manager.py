#!/usr/bin/env python3
"""
Gestionnaire de ressources pour l'évaluation optimisée des modèles

Ce script :
1. Vérifie les ressources disponibles (RAM/GPU)
2. Recommande une stratégie d'évaluation optimale
3. Lance automatiquement l'évaluation par batch
4. Surveille les ressources pendant l'exécution

Usage:
    python evaluation/code/resource_manager.py --auto --samples 30
"""

import psutil
import subprocess
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResourceManager:
    """Gestionnaire de ressources pour l'évaluation des modèles"""
    
    def __init__(self):
        self.memory_info = self.get_system_info()
        self.model_memory_estimates = self.get_model_memory_estimates()
        
    def get_system_info(self) -> Dict:
        """Récupère les informations système"""
        memory = psutil.virtual_memory()
        
        info = {
            "total_ram_gb": round(memory.total / (1024**3), 2),
            "available_ram_gb": round(memory.available / (1024**3), 2),
            "cpu_count": psutil.cpu_count(),
            "gpu_available": False
        }
        
        # Vérification GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_props = torch.cuda.get_device_properties(0)
                info.update({
                    "gpu_available": True,
                    "gpu_name": gpu_props.name,
                    "gpu_memory_gb": round(gpu_props.total_memory / (1024**3), 2)
                })
        except ImportError:
            pass
            
        return info
    
    def get_model_memory_estimates(self) -> Dict[str, float]:
        """Estimations de consommation RAM par modèle (en GB)"""
        return {
            "whisper:tiny": 1.0,
            "whisper:base": 1.0,
            "whisper:small": 2.0,
            "whisper:medium": 5.0,
            "whisper:large": 10.0,
            "whisper:large-v3": 10.0,
            "wav2vec2:xlsr-53-french": 3.0,
            "wav2vec2:bhuang-french": 2.5,
            "wav2vec2:jonatasgrosman-xlsr": 3.5,
            "wav2vec2:jonatasgrosman-voxpopuli": 3.0,
            "wav2vec2:wasertech-cv9": 2.5,
            "seamless:medium": 6.0,
            "seamless:large": 12.0,
            "gemma3n:e2b": 8.0,
            "gemma3n:e4b": 15.0,
        }
    
    def analyze_resources(self) -> Dict:
        """Analyse les ressources et recommande une stratégie"""
        analysis = {
            "total_ram": self.memory_info["total_ram_gb"],
            "available_ram": self.memory_info["available_ram_gb"],
            "safe_ram_limit": self.memory_info["available_ram_gb"] * 0.7,  # 70% de la RAM dispo
            "recommendations": []
        }
        
        # Classement des modèles par consommation
        models_by_memory = sorted(
            self.model_memory_estimates.items(),
            key=lambda x: x[1]
        )
        
        # Stratégie recommandée
        if analysis["safe_ram_limit"] > 15:
            strategy = "aggressive"
            batch_size = 4
            analysis["recommendations"].append("RAM suffisante pour des batches larges")
        elif analysis["safe_ram_limit"] > 8:
            strategy = "moderate"
            batch_size = 3
            analysis["recommendations"].append("RAM modérée - batches moyens recommandés")
        else:
            strategy = "conservative"
            batch_size = 2
            analysis["recommendations"].append("RAM limitée - petits batches obligatoires")
        
        analysis.update({
            "strategy": strategy,
            "recommended_batch_size": batch_size,
            "estimated_total_time_minutes": len(self.model_memory_estimates) * 5,  # 5min par modèle estimé
        })
        
        return analysis
    
    def create_optimized_batches(self, analysis: Dict) -> Dict[str, List[Tuple[str, str]]]:
        """Crée des batches optimisés selon l'analyse"""
        safe_limit = analysis["safe_ram_limit"]
        
        # Modèles triés par consommation RAM
        models_sorted = sorted(
            [(k.split(":")[0], k.split(":")[1], v) for k, v in self.model_memory_estimates.items()],
            key=lambda x: x[2]
        )
        
        batches = {}
        current_batch = []
        current_memory = 0
        batch_num = 1
        
        for model_type, model_size, memory_need in models_sorted:
            # Si ajouter ce modèle dépasse la limite, créer un nouveau batch
            if current_memory + memory_need > safe_limit and current_batch:
                batches[f"batch_{batch_num}"] = current_batch.copy()
                current_batch = []
                current_memory = 0
                batch_num += 1
            
            current_batch.append((model_type, model_size))
            current_memory += memory_need
        
        # Ajouter le dernier batch s'il n'est pas vide
        if current_batch:
            batches[f"batch_{batch_num}"] = current_batch
        
        return batches
    
    def display_analysis(self, analysis: Dict, batches: Dict):
        """Affiche l'analyse des ressources"""
        logger.info("\n" + "="*60)
        logger.info("🔍 ANALYSE DES RESSOURCES SYSTÈME")
        logger.info("="*60)
        logger.info(f"💾 RAM totale: {analysis['total_ram']:.1f} GB")
        logger.info(f"💾 RAM disponible: {analysis['available_ram']:.1f} GB")
        logger.info(f"🎯 Limite sécurisée: {analysis['safe_ram_limit']:.1f} GB")
        
        if self.memory_info["gpu_available"]:
            logger.info(f"🎮 GPU: {self.memory_info['gpu_name']} ({self.memory_info['gpu_memory_gb']:.1f} GB)")
        
        logger.info(f"\n📋 Stratégie recommandée: {analysis['strategy'].upper()}")
        logger.info(f"⏱️ Temps estimé: {analysis['estimated_total_time_minutes']} minutes")
        
        logger.info(f"\n📦 {len(batches)} batches optimisés:")
        for batch_name, models in batches.items():
            total_memory = sum(self.model_memory_estimates[f"{t}:{s}"] for t, s in models)
            logger.info(f"  {batch_name}: {len(models)} modèles (~{total_memory:.1f} GB)")
            for model_type, model_size in models:
                memory = self.model_memory_estimates[f"{model_type}:{model_size}"]
                logger.info(f"    - {model_type}:{model_size} (~{memory:.1f} GB)")
        
        logger.info("\n💡 Recommandations:")
        for rec in analysis["recommendations"]:
            logger.info(f"  • {rec}")
    
    def run_evaluation_with_monitoring(self, batches: Dict, num_samples: int = 30):
        """Lance l'évaluation avec monitoring automatique"""
        logger.info("\n🚀 LANCEMENT DE L'ÉVALUATION AVEC MONITORING")
        
        # Démarrage du monitoring en arrière-plan
        monitor_cmd = [
            sys.executable, "memory_monitor.py",
            "--threshold", "80",
            "--log-interval", "30"
        ]
        
        logger.info("🔍 Démarrage du monitoring mémoire...")
        monitor_process = subprocess.Popen(
            monitor_cmd,
            cwd=Path(__file__).parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            # Lancement de l'évaluation par batch
            batch_names = list(batches.keys())
            eval_cmd = [
                sys.executable, "batch_evaluation.py",
                "--samples", str(num_samples),
                "--batches"
            ] + batch_names
            
            logger.info("📊 Démarrage de l'évaluation par batch...")
            eval_result = subprocess.run(
                eval_cmd,
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=7200  # 2h timeout
            )
            
            if eval_result.returncode == 0:
                logger.info("✅ Évaluation terminée avec succès!")
                print(eval_result.stdout)
            else:
                logger.error("❌ Erreur lors de l'évaluation:")
                print(eval_result.stderr)
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Timeout de l'évaluation (2h dépassées)")
        
        finally:
            # Arrêt du monitoring
            logger.info("🛑 Arrêt du monitoring...")
            monitor_process.terminate()
            monitor_process.wait()
    
    def interactive_mode(self):
        """Mode interactif pour choisir la stratégie"""
        analysis = self.analyze_resources()
        batches = self.create_optimized_batches(analysis)
        
        self.display_analysis(analysis, batches)
        
        print("\n" + "="*60)
        print("🎯 OPTIONS D'ÉVALUATION")
        print("="*60)
        print("1. Évaluation automatique complète (recommandé)")
        print("2. Évaluation batch par batch manuelle")
        print("3. Sélection de batches spécifiques")
        print("4. Test rapide (5 échantillons)")
        print("5. Analyse seulement (pas d'évaluation)")
        
        while True:
            try:
                choice = input("\nChoisissez une option (1-5): ").strip()
                
                if choice == "1":
                    samples = int(input("Nombre d'échantillons par batch (défaut: 30): ") or "30")
                    self.run_evaluation_with_monitoring(batches, samples)
                    break
                elif choice == "2":
                    self.manual_batch_mode(batches)
                    break
                elif choice == "3":
                    self.selective_batch_mode(batches)
                    break
                elif choice == "4":
                    self.run_evaluation_with_monitoring(batches, 5)
                    break
                elif choice == "5":
                    logger.info("📊 Analyse terminée - aucune évaluation lancée")
                    break
                else:
                    print("❌ Option invalide, veuillez choisir entre 1 et 5")
                    
            except (ValueError, KeyboardInterrupt):
                print("\n🛑 Opération annulée")
                break
    
    def manual_batch_mode(self, batches: Dict):
        """Mode manuel batch par batch"""
        logger.info("🔧 Mode manuel sélectionné")
        for batch_name in batches.keys():
            response = input(f"\nExécuter {batch_name}? (o/n): ").strip().lower()
            if response in ['o', 'oui', 'y', 'yes']:
                subprocess.run([
                    sys.executable, "batch_evaluation.py",
                    "--batches", batch_name,
                    "--samples", "30"
                ], cwd=Path(__file__).parent)
    
    def selective_batch_mode(self, batches: Dict):
        """Mode de sélection de batches spécifiques"""
        print("\n📦 Batches disponibles:")
        for i, batch_name in enumerate(batches.keys(), 1):
            print(f"  {i}. {batch_name}")
        
        selected = input("\nNuméros des batches à exécuter (ex: 1,3,5): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in selected.split(",")]
            selected_batches = [list(batches.keys())[i] for i in indices]
            
            subprocess.run([
                sys.executable, "batch_evaluation.py",
                "--batches"
            ] + selected_batches + [
                "--samples", "30"
            ], cwd=Path(__file__).parent)
            
        except (ValueError, IndexError):
            logger.error("❌ Sélection invalide")

def main():
    parser = argparse.ArgumentParser(description="Gestionnaire de ressources pour l'évaluation")
    parser.add_argument('--auto', action='store_true',
                       help="Mode automatique sans interaction")
    parser.add_argument('--samples', type=int, default=30,
                       help="Nombre d'échantillons par batch")
    parser.add_argument('--analyze-only', action='store_true',
                       help="Analyse seulement, pas d'évaluation")
    
    args = parser.parse_args()
    
    manager = ResourceManager()
    
    if args.analyze_only:
        analysis = manager.analyze_resources()
        batches = manager.create_optimized_batches(analysis)
        manager.display_analysis(analysis, batches)
    elif args.auto:
        analysis = manager.analyze_resources()
        batches = manager.create_optimized_batches(analysis)
        manager.display_analysis(analysis, batches)
        manager.run_evaluation_with_monitoring(batches, args.samples)
    else:
        manager.interactive_mode()

if __name__ == "__main__":
    main() 