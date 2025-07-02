#!/usr/bin/env python3
"""
Script d'analyse détaillée des erreurs pour les modèles Whisper

Ce script :
1. Lit les résultats détaillés générés par evaluate_whisper_models.py
2. Effectue une analyse WER détaillée par clip
3. Génère des rapports d'erreurs par catégorie
4. Crée des graphiques de répartition des erreurs

Usage:
    python evaluation/code/detailed_error_analysis.py [--results-file detailed_results_*.json]
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
import glob
import argparse
from typing import Dict, List, Optional
from collections import defaultdict

# Ajout du chemin racine pour importer les modules
sys.path.append(str(Path(__file__).parent.parent.parent))
from metrics import SpeechRecognitionMetrics

# Configuration matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DetailedErrorAnalyzer:
    """Analyseur détaillé des erreurs pour les évaluations Whisper"""
    
    def __init__(self, output_dir: str = "evaluation/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = SpeechRecognitionMetrics()
        
        # Stockage des analyses
        self.error_analysis = {}
        self.summary_stats = {}
        
    def load_detailed_results(self, results_file: Optional[str] = None) -> List[Dict]:
        """Charge les résultats détaillés d'évaluation"""
        if results_file:
            results_path = Path(results_file)
        else:
            # Chercher le fichier le plus récent
            pattern = str(self.output_dir / "detailed_results_*.json")
            result_files = glob.glob(pattern)
            if not result_files:
                raise FileNotFoundError("Aucun fichier de résultats détaillés trouvé")
            results_path = Path(max(result_files))
        
        print(f"📖 Chargement des résultats détaillés : {results_path}")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_single_prediction(self, reference: str, prediction: str) -> Dict:
        """Analyse détaillée d'une prédiction individuelle"""
        # Analyse WER détaillée
        wer_detail = self.metrics.detailed_wer_analysis(reference, prediction, normalize=True)
        
        # Métriques supplémentaires
        cer = self.metrics.calculate_cer(reference, prediction)
        
        return {
            'wer': wer_detail['wer'],
            'substitutions': wer_detail['substitutions'],
            'insertions': wer_detail['insertions'],
            'deletions': wer_detail['deletions'],
            'reference_words': len(reference.split()),
            'prediction_words': len(prediction.split()),
            'cer': cer,
            'reference_chars': len(reference),
            'prediction_chars': len(prediction)
        }
    
    def analyze_all_results(self, detailed_results: List[Dict]) -> Dict:
        """Analyse tous les résultats par modèle"""
        print("🔍 Analyse détaillée des erreurs en cours...")
        
        # Grouper par modèle
        models_data = defaultdict(list)
        for result in detailed_results:
            models_data[result['model']].append(result)
        
        analysis_by_model = {}
        
        for model, model_results in models_data.items():
            print(f"  📊 Analyse du modèle {model}...")
            
            model_analysis = []
            total_errors = {'substitutions': 0, 'insertions': 0, 'deletions': 0}
            total_wer = 0
            total_cer = 0
            
            for result in model_results:
                analysis = self.analyze_single_prediction(
                    result['reference'], 
                    result['prediction']
                )
                
                # Ajouter les métadonnées
                analysis.update({
                    'file_id': result['file_id'],
                    'reference': result['reference'],
                    'prediction': result['prediction'],
                    'inference_time': result.get('inference_time', 0)
                })
                
                model_analysis.append(analysis)
                
                # Accumulation pour les statistiques
                total_errors['substitutions'] += analysis['substitutions']
                total_errors['insertions'] += analysis['insertions']
                total_errors['deletions'] += analysis['deletions']
                total_wer += analysis['wer']
                total_cer += analysis['cer']
            
            # Statistiques du modèle
            num_samples = len(model_results)
            analysis_by_model[model] = {
                'detailed_analysis': model_analysis,
                'summary': {
                    'avg_wer': total_wer / num_samples,
                    'avg_cer': total_cer / num_samples,
                    'total_substitutions': total_errors['substitutions'],
                    'total_insertions': total_errors['insertions'],
                    'total_deletions': total_errors['deletions'],
                    'avg_substitutions_per_clip': total_errors['substitutions'] / num_samples,
                    'avg_insertions_per_clip': total_errors['insertions'] / num_samples,
                    'avg_deletions_per_clip': total_errors['deletions'] / num_samples,
                    'num_samples': num_samples
                }
            }
        
        self.error_analysis = analysis_by_model
        return analysis_by_model
    
    def generate_error_report(self, model: str, max_examples: int = 10):
        """Génère un rapport textuel des erreurs pour un modèle"""
        if model not in self.error_analysis:
            print(f"❌ Modèle {model} non trouvé dans l'analyse")
            return
        
        model_data = self.error_analysis[model]
        summary = model_data['summary']
        detailed = model_data['detailed_analysis']
        
        print(f"\\n{'='*80}")
        print(f"📊 RAPPORT D'ERREURS DÉTAILLÉ - MODÈLE {model.upper()}")
        print(f"{'='*80}")
        
        print(f"\\n📈 STATISTIQUES GLOBALES:")
        print(f"  Nombre d'échantillons    : {summary['num_samples']}")
        print(f"  WER moyen               : {summary['avg_wer']:.2f}%")
        print(f"  CER moyen               : {summary['avg_cer']:.2f}%")
        
        print(f"\\n🔢 RÉPARTITION DES ERREURS:")
        print(f"  Total substitutions     : {summary['total_substitutions']}")
        print(f"  Total insertions        : {summary['total_insertions']}")
        print(f"  Total suppressions      : {summary['total_deletions']}")
        
        print(f"\\n📊 MOYENNES PAR CLIP:")
        print(f"  Substitutions/clip      : {summary['avg_substitutions_per_clip']:.2f}")
        print(f"  Insertions/clip         : {summary['avg_insertions_per_clip']:.2f}")
        print(f"  Suppressions/clip       : {summary['avg_deletions_per_clip']:.2f}")
        
        # Exemples des pires erreurs
        worst_errors = sorted(detailed, key=lambda x: x['wer'], reverse=True)[:max_examples]
        
        print(f"\\n🚨 {max_examples} PIRES ERREURS (WER le plus élevé):")
        print("-" * 80)
        
        for i, error in enumerate(worst_errors, 1):
            print(f"\\n{i}. {error['file_id']} (WER: {error['wer']:.1f}%)")
            print(f"   Référence  : {error['reference']}")
            print(f"   Prédiction : {error['prediction']}")
            print(f"   Erreurs    : {error['substitutions']}S + {error['insertions']}I + {error['deletions']}D")
    
    def create_error_distribution_plots(self, timestamp: str = ""):
        """Crée des graphiques de distribution des erreurs"""
        if not self.error_analysis:
            print("❌ Aucune analyse d'erreur disponible")
            return
        
        models = list(self.error_analysis.keys())
        
        # Préparation des données
        error_data = []
        for model in models:
            summary = self.error_analysis[model]['summary']
            error_data.append({
                'Model': model,
                'Substitutions': summary['avg_substitutions_per_clip'],
                'Insertions': summary['avg_insertions_per_clip'],
                'Suppressions': summary['avg_deletions_per_clip']
            })
        
        df_errors = pd.DataFrame(error_data)
        
        # Graphique 1: Distribution des types d'erreurs
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Barres empilées
        df_plot = df_errors.set_index('Model')[['Substitutions', 'Insertions', 'Suppressions']]
        df_plot.plot(kind='bar', stacked=True, ax=ax1, color=['#e74c3c', '#f39c12', '#3498db'])
        ax1.set_title('Répartition des Types d\'Erreurs par Modèle', fontweight='bold')
        ax1.set_ylabel('Nombre moyen d\'erreurs par clip')
        ax1.set_xlabel('Modèle Whisper')
        ax1.legend(title='Type d\'erreur')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Graphique en barres séparées
        x = np.arange(len(models))
        width = 0.25
        
        ax2.bar(x - width, df_errors['Substitutions'], width, label='Substitutions', color='#e74c3c', alpha=0.8)
        ax2.bar(x, df_errors['Insertions'], width, label='Insertions', color='#f39c12', alpha=0.8)
        ax2.bar(x + width, df_errors['Suppressions'], width, label='Suppressions', color='#3498db', alpha=0.8)
        
        ax2.set_title('Comparaison des Types d\'Erreurs', fontweight='bold')
        ax2.set_ylabel('Nombre moyen d\'erreurs par clip')
        ax2.set_xlabel('Modèle Whisper')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarde
        output_file = self.output_dir / f"error_distribution{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Graphique sauvegardé : {output_file}")
        plt.close()
    
    def create_wer_distribution_plot(self, timestamp: str = ""):
        """Crée un graphique de distribution des WER par clip"""
        if not self.error_analysis:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        models = list(self.error_analysis.keys())
        
        for i, model in enumerate(models):
            if i >= len(axes):
                break
                
            ax = axes[i]
            detailed = self.error_analysis[model]['detailed_analysis']
            wer_values = [item['wer'] for item in detailed]
            
            # Histogramme
            ax.hist(wer_values, bins=20, alpha=0.7, color=f'C{i}', edgecolor='black')
            ax.set_title(f'Distribution WER - {model}', fontweight='bold')
            ax.set_xlabel('WER (%)')
            ax.set_ylabel('Nombre de clips')
            ax.grid(axis='y', alpha=0.3)
            
            # Statistiques
            mean_wer = np.mean(wer_values)
            median_wer = np.median(wer_values)
            ax.axvline(mean_wer, color='red', linestyle='--', label=f'Moyenne: {mean_wer:.1f}%')
            ax.axvline(median_wer, color='orange', linestyle='--', label=f'Médiane: {median_wer:.1f}%')
            ax.legend()
        
        # Masquer les axes inutilisés
        for i in range(len(models), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Sauvegarde
        output_file = self.output_dir / f"wer_distribution{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Graphique sauvegardé : {output_file}")
        plt.close()
    
    def save_detailed_analysis(self, timestamp: str = ""):
        """Sauvegarde l'analyse détaillée en JSON et CSV"""
        if not self.error_analysis:
            return
        
        # Sauvegarde JSON complète
        json_file = self.output_dir / f"detailed_error_analysis{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.error_analysis, f, indent=2, ensure_ascii=False)
        
        # Sauvegarde CSV des résumés
        summary_data = []
        for model, data in self.error_analysis.items():
            summary = data['summary']
            summary['model'] = model
            summary_data.append(summary)
        
        df_summary = pd.DataFrame(summary_data)
        csv_file = self.output_dir / f"error_summary{timestamp}.csv"
        df_summary.to_csv(csv_file, index=False)
        
        print(f"💾 Analyse détaillée sauvegardée :")
        print(f"  - {json_file}")
        print(f"  - {csv_file}")
    
    def run_full_analysis(self, results_file: Optional[str] = None):
        """Lance l'analyse complète"""
        from datetime import datetime
        timestamp = "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Chargement des données
        detailed_results = self.load_detailed_results(results_file)
        
        # Analyse
        self.analyze_all_results(detailed_results)
        
        # Génération des rapports
        print("\\n" + "="*80)
        print("📊 GÉNÉRATION DES RAPPORTS D'ERREURS")
        print("="*80)
        
        for model in self.error_analysis.keys():
            self.generate_error_report(model, max_examples=5)
        
        # Génération des graphiques
        print("\\n🎨 Génération des graphiques...")
        self.create_error_distribution_plots(timestamp)
        self.create_wer_distribution_plot(timestamp)
        
        # Sauvegarde
        self.save_detailed_analysis(timestamp)
        
        print("\\n✅ Analyse détaillée terminée !")


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Analyse détaillée des erreurs Whisper")
    parser.add_argument("--results-file", "-r", type=str,
                       help="Fichier de résultats détaillés spécifique (optionnel)")
    
    args = parser.parse_args()
    
    try:
        analyzer = DetailedErrorAnalyzer()
        analyzer.run_full_analysis(args.results_file)
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 