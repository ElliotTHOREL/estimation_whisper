#!/usr/bin/env python3
"""
Script d'évaluation par batch pour optimiser l'utilisation de la RAM

Ce script :
1. Organise les modèles en batches selon leur consommation RAM
2. Évalue chaque batch séparément
3. Vide le cache entre les batches
4. Fusionne les résultats finaux
5. Génère les graphiques de comparaison

Usage:
    python evaluation/code/batch_evaluation.py --batch-size 3 --samples 30
"""

import sys
import time
import json
import requests
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from datetime import datetime
import argparse
import glob

# Ajout du chemin racine
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchEvaluator:
    """Gestionnaire d'évaluation par batch avec optimisation mémoire"""
    
    def __init__(self, api_url: str = "http://localhost:8000", output_dir: str = "evaluation/benchmarks"):
        self.api_url = api_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Organisation des modèles par batch (selon consommation RAM estimée)
        self.model_batches = {
            "batch_1_light": [
                ("whisper", "tiny"),
                ("whisper", "base"),
                ("whisper", "small"),
            ],
            "batch_2_medium": [
                ("whisper", "medium"),
                ("wav2vec2", "xlsr-53-french"),
                ("wav2vec2", "bhuang-french"),
            ],
            "batch_3_heavy": [
                ("whisper", "large"),
                ("whisper", "large-v3"),
            ],
            "batch_4_wav2vec": [
                ("wav2vec2", "jonatasgrosman-xlsr"),
                ("wav2vec2", "jonatasgrosman-voxpopuli"),
                ("wav2vec2", "wasertech-cv9"),
            ],
            "batch_5_seamless": [
                ("seamless", "medium"),
                ("seamless", "large"),
            ],
            "batch_6_gemma": [
                ("gemma3n", "e2b"),
                ("gemma3n", "e4b"),
            ]
        }
        
        self.all_results = {}
        self.all_detailed_results = []
    
    def check_api_health(self) -> bool:
        """Vérifie que l'API est accessible"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"API non accessible : {e}")
            return False
    
    def clear_api_cache(self) -> bool:
        """Vide le cache de l'API pour libérer la mémoire"""
        try:
            logger.info("🧹 Vidage du cache API...")
            response = requests.post(f"{self.api_url}/cache/clear", timeout=30)
            if response.status_code == 200:
                logger.info("✅ Cache vidé avec succès")
                time.sleep(5)  # Pause pour s'assurer que la mémoire est libérée
                return True
            else:
                logger.warning(f"⚠️ Échec du vidage de cache: {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors du vidage de cache: {e}")
            return False
    
    def evaluate_single_batch(self, batch_name: str, models: List[Tuple[str, str]], 
                            num_samples: int = 30) -> Dict:
        """Évalue un batch de modèles"""
        logger.info(f"🚀 Début de l'évaluation du {batch_name}")
        logger.info(f"   Modèles: {[f'{t}:{s}' for t, s in models]}")
        
        # Préparation de la commande d'évaluation
        models_str = ",".join([f"{model_type}:{model_size}" for model_type, model_size in models])
        
        # Import du module d'évaluation
        from evaluation_speech_models import SpeechAPIEvaluator
        
        try:
            # Lancement de l'évaluation pour ce batch
            evaluator = SpeechAPIEvaluator(
                api_url=self.api_url,
                output_dir=str(self.output_dir),
                models=models
            )
            
            batch_results = evaluator.run_evaluation(num_samples=num_samples)
            
            # Sauvegarde des résultats du batch
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_file = self.output_dir / f"{batch_name}_results_{timestamp}.json"
            
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ {batch_name} terminé - {len(batch_results)} modèles évalués")
            logger.info(f"💾 Résultats sauvegardés: {batch_file}")
            
            # Ajout aux résultats globaux
            self.all_results.update(batch_results)
            self.all_detailed_results.extend(evaluator.detailed_results)
            
            return batch_results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'évaluation du {batch_name}: {e}")
            return {}
    
    def run_batch_evaluation(self, num_samples: int = 30, selected_batches: List[str] = None):
        """Lance l'évaluation complète par batches"""
        if not self.check_api_health():
            raise RuntimeError("❌ API Speech non accessible")
        
        # Sélection des batches à évaluer
        if selected_batches:
            batches_to_run = {k: v for k, v in self.model_batches.items() if k in selected_batches}
        else:
            batches_to_run = self.model_batches
        
        logger.info(f"🎯 Évaluation par batch - {len(batches_to_run)} batches à traiter")
        logger.info(f"📊 Échantillons par batch: {num_samples}")
        
        start_time = time.time()
        
        for i, (batch_name, models) in enumerate(batches_to_run.items(), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"📦 BATCH {i}/{len(batches_to_run)}: {batch_name}")
            logger.info(f"{'='*60}")
            
            # Évaluation du batch
            batch_results = self.evaluate_single_batch(batch_name, models, num_samples)
            
            if batch_results:
                # Affichage des résultats du batch
                self.display_batch_summary(batch_name, batch_results)
            
            # Vidage du cache entre les batches (sauf pour le dernier)
            if i < len(batches_to_run):
                self.clear_api_cache()
                logger.info(f"⏱️ Pause de 10s avant le batch suivant...")
                time.sleep(10)
        
        total_time = time.time() - start_time
        logger.info(f"\n🏁 Évaluation complète terminée en {total_time/60:.1f} minutes")
        
        # Sauvegarde des résultats consolidés
        self.save_consolidated_results()
        
        return self.all_results
    
    def display_batch_summary(self, batch_name: str, results: Dict):
        """Affiche un résumé des résultats du batch"""
        logger.info(f"\n📊 Résumé du {batch_name}:")
        for model, metrics in results.items():
            wer = metrics.get('wer', 0)
            bleu = metrics.get('bleu', 0)
            time_avg = metrics.get('avg_inference_time', 0)
            logger.info(f"  {model:25} | WER: {wer:6.3f} | BLEU: {bleu:6.3f} | Temps: {time_avg:6.3f}s")
    
    def aggregate_all_batch_results(self):
        """Agréger tous les fichiers batchs *_results_*.json du dossier en un seul fichier global"""
        result_files = glob.glob(str(self.output_dir / "*_results_*.json"))
        all_results = {}
        for file in result_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    all_results.update(data)
        output = self.output_dir / "consolidated_results_global.json"
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"🌍 Résultats globaux agrégés dans {output}")

        # Agrégation des fichiers détaillés
        detailed_files = glob.glob(str(self.output_dir / "*detailed_results_*.json"))
        all_detailed = []
        for file in detailed_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_detailed.extend(data)
        output_detailed = self.output_dir / "consolidated_detailed_global.json"
        with open(output_detailed, 'w', encoding='utf-8') as f:
            json.dump(all_detailed, f, indent=2, ensure_ascii=False)
        logger.info(f"🌍 Résultats détaillés globaux agrégés dans {output_detailed}")

    def save_consolidated_results(self):
        """Sauvegarde les résultats consolidés de tous les batches du run courant et met à jour le global"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Résultats consolidés du run courant
        consolidated_file = self.output_dir / f"consolidated_results_{timestamp}.json"
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, indent=2, ensure_ascii=False)
        # Résultats détaillés du run courant
        detailed_file = self.output_dir / f"consolidated_detailed_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_detailed_results, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Résultats consolidés sauvegardés:")
        logger.info(f"   📊 Métriques: {consolidated_file}")
        logger.info(f"   🔍 Détails: {detailed_file}")
        # Génération automatique des graphiques
        self.generate_final_plots(str(consolidated_file))
        # Agrégation globale
        self.aggregate_all_batch_results()
    
    def generate_final_plots(self, results_file: str):
        """Génère les graphiques finaux avec tous les résultats"""
        logger.info("🎨 Génération des graphiques consolidés...")
        
        try:
            from generate_plots import PlotGenerator
            plot_gen = PlotGenerator(output_dir=str(self.output_dir))
            plot_gen.generate_all_plots(results_file=results_file)
            logger.info("✅ Graphiques générés avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération des graphiques: {e}")
    
    def list_available_batches(self):
        """Affiche les batches disponibles"""
        logger.info("📦 Batches disponibles:")
        for batch_name, models in self.model_batches.items():
            logger.info(f"  {batch_name}:")
            for model_type, model_size in models:
                logger.info(f"    - {model_type}:{model_size}")

def main():
    parser = argparse.ArgumentParser(description="Évaluation par batch des modèles Speech-to-Text")
    parser.add_argument('--api-url', type=str, default="http://localhost:8000", 
                       help="URL de l'API Speech-to-Text")
    parser.add_argument('--samples', type=int, default=30, 
                       help="Nombre d'échantillons par batch")
    parser.add_argument('--batches', type=str, nargs='+', 
                       help="Batches spécifiques à évaluer (ex: batch_1_light batch_2_medium)")
    parser.add_argument('--list-batches', action='store_true', 
                       help="Afficher les batches disponibles")
    parser.add_argument('--output-dir', type=str, default="evaluation/benchmarks",
                       help="Dossier de sortie pour les résultats")
    
    args = parser.parse_args()
    
    evaluator = BatchEvaluator(api_url=args.api_url, output_dir=args.output_dir)
    
    if args.list_batches:
        evaluator.list_available_batches()
        return
    
    # Lancement de l'évaluation
    evaluator.run_batch_evaluation(
        num_samples=args.samples,
        selected_batches=args.batches
    )

if __name__ == "__main__":
    main() 