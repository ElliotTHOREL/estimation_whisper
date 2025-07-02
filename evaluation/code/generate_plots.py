#!/usr/bin/env python3
"""
Script pour générer les graphiques d'évaluation des modèles Whisper et Wav2Vec2

Ce script :
1. Charge les résultats d'évaluation
2. Génère des graphiques comparatifs
3. Sauvegarde les plots dans le dossier benchmarks

Usage:
    python evaluation/code/generate_plots.py [results_file.json]
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
import glob
from typing import Dict, Optional
import argparse

# Configuration matplotlib pour de beaux graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PlotGenerator:
    """Générateur de graphiques pour l'évaluation Whisper"""
    
    def __init__(self, output_dir: str = "evaluation/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration des graphiques
        self.figsize = (12, 8)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
    def load_latest_results(self, results_file: Optional[str] = None) -> Dict:
        """Charge les derniers résultats d'évaluation"""
        if results_file:
            results_path = Path(results_file)
        else:
            # Chercher le fichier le plus récent
            pattern = str(self.output_dir / "evaluation_results_*.json")
            result_files = glob.glob(pattern)
            if not result_files:
                raise FileNotFoundError("Aucun fichier de résultats trouvé")
            results_path = Path(max(result_files))
        
        print(f"📖 Chargement des résultats : {results_path}")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_metrics_comparison(self, results: Dict, timestamp: str = ""):
        """Crée un graphique de comparaison des métriques principales"""
        models = list(results.keys())
        metrics = ['wer', 'cer', 'bleu', 'mer']
        
        # Préparation des données
        data = []
        for model in models:
            for metric in metrics:
                value = results[model].get(metric, 0)
                data.append({
                    'Model': model,
                    'Metric': metric.upper(),
                    'Value': value
                })
        
        df = pd.DataFrame(data)
        
        # Création du graphique
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparaison des Métriques Whisper', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            metric_data = df[df['Metric'] == metric.upper()]
            
            bars = ax.bar(metric_data['Model'], metric_data['Value'], 
                         color=self.colors[:len(models)], alpha=0.8)
            
            # Personnalisation
            ax.set_title(f'{metric.upper()} par Modèle', fontweight='bold')
            ax.set_ylabel(f'{metric.upper()} Score')
            ax.set_xlabel('Modèle Whisper')
            
            # Rotation des labels si nécessaire
            if len(models) > 4:
                ax.tick_params(axis='x', rotation=45)
            
            # Ajout des valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Amélioration visuelle
            ax.grid(axis='y', alpha=0.3)
            ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Sauvegarde
        output_file = self.output_dir / f"metrics_comparison{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Graphique sauvegardé : {output_file}")
        plt.close()
    
    def create_performance_vs_accuracy(self, results: Dict, timestamp: str = ""):
        """Crée un graphique performance vs précision"""
        models = list(results.keys())
        
        # Extraction des données
        wer_scores = [results[model].get('wer', 0) for model in models]
        inference_times = [results[model].get('avg_inference_time', 0) for model in models]
        bleu_scores = [results[model].get('bleu', 0) for model in models]
        
        # Création du graphique
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Graphique 1: WER vs Temps d'inférence
        scatter1 = ax1.scatter(inference_times, wer_scores, 
                              s=200, c=self.colors[:len(models)], alpha=0.7)
        
        for i, model in enumerate(models):
            ax1.annotate(model, (inference_times[i], wer_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontweight='bold')
        
        ax1.set_xlabel('Temps d\'inférence moyen (s)', fontweight='bold')
        ax1.set_ylabel('WER (plus bas = meilleur)', fontweight='bold')
        ax1.set_title('Compromis Vitesse vs Précision (WER)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: BLEU vs Temps d'inférence
        scatter2 = ax2.scatter(inference_times, bleu_scores, 
                              s=200, c=self.colors[:len(models)], alpha=0.7)
        
        for i, model in enumerate(models):
            ax2.annotate(model, (inference_times[i], bleu_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontweight='bold')
        
        ax2.set_xlabel('Temps d\'inférence moyen (s)', fontweight='bold')
        ax2.set_ylabel('BLEU Score (plus haut = meilleur)', fontweight='bold')
        ax2.set_title('Compromis Vitesse vs Qualité (BLEU)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarde
        output_file = self.output_dir / f"performance_vs_accuracy{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Graphique sauvegardé : {output_file}")
        plt.close()
    
    def create_radar_chart(self, results: Dict, timestamp: str = ""):
        """Crée un radar chart pour comparer les modèles"""
        models = list(results.keys())
        metrics = ['wer', 'cer', 'bleu', 'mer', 'success_rate']
        
        # Normalisation des métriques (pour le radar chart)
        normalized_data = {}
        
        for model in models:
            normalized_data[model] = []
            for metric in metrics:
                value = results[model].get(metric, 0)
                
                # Normalisation inverse pour WER, CER, MER (plus bas = meilleur)
                if metric in ['wer', 'cer', 'mer']:
                    # Convertir en "score de qualité" (1 - metric normalisé)
                    max_val = max([results[m].get(metric, 0) for m in models])
                    if max_val > 0:
                        normalized_value = 1 - (value / max_val)
                    else:
                        normalized_value = 1
                else:
                    # Pour BLEU et success_rate (plus haut = meilleur)
                    max_val = max([results[m].get(metric, 0) for m in models])
                    if max_val > 0:
                        normalized_value = value / max_val
                    else:
                        normalized_value = 0
                
                normalized_data[model].append(normalized_value)
        
        # Création du radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Fermer le cercle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for i, model in enumerate(models):
            values = normalized_data[model] + [normalized_data[model][0]]  # Fermer le cercle
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=self.colors[i])
            ax.fill(angles, values, alpha=0.25, color=self.colors[i])
        
        # Personnalisation
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Comparaison Globale des Modèles Whisper', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        
        # Sauvegarde
        output_file = self.output_dir / f"radar_comparison{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Graphique sauvegardé : {output_file}")
        plt.close()
    
    def create_summary_table(self, results: Dict, timestamp: str = ""):
        """Crée un tableau récapitulatif des résultats"""
        models = list(results.keys())
        
        # Création du DataFrame
        data = []
        for model in models:
            metrics = results[model]
            data.append({
                'Modèle': model,
                'WER': f"{metrics.get('wer', 0):.3f}",
                'CER': f"{metrics.get('cer', 0):.3f}",
                'BLEU': f"{metrics.get('bleu', 0):.3f}",
                'MER': f"{metrics.get('mer', 0):.3f}",
                'Temps (s)': f"{metrics.get('avg_inference_time', 0):.2f}",
                'Succès (%)': f"{metrics.get('success_rate', 0):.1f}"
            })
        
        df = pd.DataFrame(data)
        
        # Création du graphique tableau
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Création du tableau
        table = ax.table(cellText=df.values,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center')
        
        # Styling du tableau
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Coloration de l'en-tête
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternance des couleurs des lignes
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F2F2F2')
        
        plt.title('Tableau Récapitulatif - Évaluation Modèles Whisper', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Sauvegarde
        output_file = self.output_dir / f"summary_table{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Tableau sauvegardé : {output_file}")
        plt.close()
    
    def generate_all_plots(self, results_file: Optional[str] = None):
        """Génère tous les graphiques d'évaluation"""
        print("🎨 Génération des graphiques d'évaluation...")
        
        # Chargement des résultats
        results = self.load_latest_results(results_file)
        
        if not results:
            print("❌ Aucun résultat à traiter")
            return
        
        # Timestamp pour les fichiers
        from datetime import datetime
        timestamp = "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"📊 Génération des graphiques pour {len(results)} modèles...")
        
        # Génération de tous les graphiques
        self.create_metrics_comparison(results, timestamp)
        self.create_performance_vs_accuracy(results, timestamp)
        self.create_radar_chart(results, timestamp)
        self.create_summary_table(results, timestamp)
        
        print("✅ Tous les graphiques ont été générés !")
        print(f"📁 Dossier de sortie : {self.output_dir}")


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Génère des graphiques d'évaluation Whisper")
    parser.add_argument("--results-file", "-r", type=str, 
                       help="Fichier de résultats spécifique (optionnel)")
    
    args = parser.parse_args()
    
    try:
        generator = PlotGenerator()
        generator.generate_all_plots(args.results_file)
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération des graphiques : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 