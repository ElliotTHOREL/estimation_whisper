#!/usr/bin/env python3
# Anciennement evaluate_whisper_models.py
"""
Script d'évaluation complète des modèles Speech-to-Text via API universelle

Ce script :
1. Utilise l'API universelle pour transcrire un échantillon d'audio
2. Compare les transcriptions avec les références
3. Calcule les métriques (WER, CER, BLEU, MER)
4. Génère des graphiques de comparaison
5. Sauvegarde les résultats dans le dossier benchmarks

Usage:
    python evaluation/code/evaluate_whisper_models.py --models whisper:base,wav2vec2:xlsr-53-french
"""

import requests
import json
import time
import pandas as pd
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import argparse

# Ajout du chemin racine pour importer les modules
sys.path.append(str(Path(__file__).parent.parent.parent))
from metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_models_arg(models_arg: str) -> List[Tuple[str, str]]:
    models = []
    for m in models_arg.split(","):
        if ":" in m:
            t, s = m.split(":", 1)
            models.append((t.strip(), s.strip()))
    return models

class SpeechAPIEvaluator:
    def __init__(self, api_url: str = "http://localhost:8000", output_dir: str = "evaluation/benchmarks", models: Optional[List[Tuple[str, str]]] = None):
        self.api_url = api_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models = models or [("whisper", "base")]
        self.results = {}
        self.detailed_results = []
    
    def check_api_health(self) -> bool:
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"API non accessible : {e}")
            return False
    
    def load_test_dataset(self, data_path: str = "cv-corpus-21.0-delta-2025-03-14/fr") -> List[Dict]:
        try:
            evaluation_dir = Path(__file__).parent.parent
            validated_path = evaluation_dir / data_path / "validated.tsv"
            if not validated_path.exists():
                logger.warning(f"Fichier validated.tsv introuvable, utilisation d'other.tsv")
                validated_path = evaluation_dir / data_path / "other.tsv"
            df = pd.read_csv(validated_path, sep='\t')
            sample_size = min(50, len(df))
            df_sample = df.head(sample_size)
            test_data = []
            clips_dir = evaluation_dir / data_path / "clips"
            for _, row in df_sample.iterrows():
                audio_path = (clips_dir / row['path']).resolve()
                if audio_path.exists():
                    test_data.append({
                        'audio_path': str(audio_path),
                        'reference': row['sentence'],
                        'file_id': row['path']
                    })
            logger.info(f"Dataset chargé : {len(test_data)} échantillons")
            return test_data
        except Exception as e:
            logger.error(f"Erreur lors du chargement du dataset : {e}")
            return []
    
    def transcribe_with_model(self, audio_path: str, model_type: str, model_size: str, language: str = "fr") -> Optional[Dict]:
        try:
            data = {
                'file_path': audio_path,
                'model_type': model_type,
                'model_size': model_size,
                'language': language,
                'return_timestamps': 'false'
            }
            response = requests.post(f"{self.api_url}/transcribe_file", data=data, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Erreur API {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Erreur lors de la transcription avec {model_type}:{model_size}: {e}")
            return None
    
    def evaluate_single_model(self, model_type: str, model_size: str, test_data: List[Dict]) -> Dict:
        logger.info(f"🔄 Évaluation du modèle {model_type}:{model_size}...")
        predictions = []
        references = []
        inference_times = []
        errors = 0
        for i, item in enumerate(test_data):
            logger.info(f"  Échantillon {i+1}/{len(test_data)}: {item['file_id']}")
            result = self.transcribe_with_model(item['audio_path'], model_type, model_size)
            if result and 'text' in result:
                predictions.append(result['text'])
                references.append(item['reference'])
                inference_times.append(result.get('inference_time', 0))
                self.detailed_results.append({
                    'model': f"{model_type}:{model_size}",
                    'file_id': item['file_id'],
                    'reference': item['reference'],
                    'prediction': result['text'],
                    'inference_time': result.get('inference_time', 0)
                })
            else:
                errors += 1
                logger.warning(f"  Échec de transcription pour {item['file_id']}")
        if not predictions:
            logger.error(f"Aucune transcription réussie pour {model_type}:{model_size}")
            return {}
        metrics = calculate_metrics(references, predictions)
        metrics.update({
            'avg_inference_time': sum(inference_times) / len(inference_times),
            'total_inference_time': sum(inference_times),
            'successful_transcriptions': len(predictions),
            'failed_transcriptions': errors,
            'success_rate': len(predictions) / (len(predictions) + errors) * 100
        })
        logger.info(f"✅ {model_type}:{model_size} - WER: {metrics['wer']:.3f}, CER: {metrics['cer']:.3f}, BLEU: {metrics['bleu']:.3f}")
        return metrics
    
    def run_evaluation(self, num_samples: int = 50) -> Dict:
        if not self.check_api_health():
            raise RuntimeError("API Speech non accessible")
        logger.info("🚀 Début de l'évaluation des modèles Speech-to-Text")
        test_data = self.load_test_dataset()
        if not test_data:
            raise RuntimeError("Impossible de charger le dataset de test")
        test_data = test_data[:num_samples]
        logger.info(f"Évaluation sur {len(test_data)} échantillons")
        start_time = time.time()
        for model_type, model_size in self.models:
            try:
                model_metrics = self.evaluate_single_model(model_type, model_size, test_data)
                if model_metrics:
                    self.results[f"{model_type}:{model_size}"] = model_metrics
            except Exception as e:
                logger.error(f"Erreur lors de l'évaluation de {model_type}:{model_size}: {e}")
        total_time = time.time() - start_time
        logger.info(f"🏁 Évaluation terminée en {total_time:.1f}s")
        self.save_results()
        return self.results
    
    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        detailed_file = self.output_dir / f"detailed_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(self.detailed_results, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Résultats sauvegardés : {results_file}")
        logger.info(f"💾 Résultats détaillés : {detailed_file}")

def main():
    parser = argparse.ArgumentParser(description="Évaluation multi-modèles Speech-to-Text via API universelle")
    parser.add_argument('--api-url', type=str, default="http://localhost:8000", help="URL de l'API Speech-to-Text")
    parser.add_argument('--models', type=str, default="whisper:base,wav2vec2:xlsr-53-french", help="Liste des modèles à tester, ex: whisper:base,wav2vec2:xlsr-53-french")
    parser.add_argument('--num-samples', type=int, default=50, help="Nombre d'échantillons à évaluer")
    args = parser.parse_args()
    models = parse_models_arg(args.models)
    evaluator = SpeechAPIEvaluator(api_url=args.api_url, models=models)
    evaluator.run_evaluation(num_samples=args.num_samples)

if __name__ == "__main__":
    main() 