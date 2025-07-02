#!/usr/bin/env python3
"""
Script principal pour lancer l'évaluation complète des modèles Speech-to-Text (Whisper, Wav2Vec2)

Ce script :
1. Lance l'évaluation de tous les modèles
2. Génère automatiquement tous les graphiques
3. Crée un rapport HTML récapitulatif

Usage:
    python evaluation/code/run_full_evaluation.py [--samples N]
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_api_status():
    """Vérifie que l'API est accessible"""
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def run_evaluation(num_samples: int = 30):
    """Lance l'évaluation des modèles"""
    logger.info("🚀 Lancement de l'évaluation des modèles Speech-to-Text...")
    
    # Script d'évaluation dans le même dossier
    evaluation_script = Path("evaluate_speech_models.py")
    
    try:
        # Lancement du script d'évaluation
        cmd = [sys.executable, str(evaluation_script)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1h timeout
        
        if result.returncode == 0:
            logger.info("✅ Évaluation terminée avec succès")
            return True
        else:
            logger.error(f"❌ Erreur lors de l'évaluation: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout lors de l'évaluation (> 1h)")
        return False
    except Exception as e:
        logger.error(f"❌ Erreur inattendue: {e}")
        return False


def generate_plots():
    """Génère tous les graphiques"""
    logger.info("🎨 Génération des graphiques...")
    
    plots_script = Path("generate_plots.py")
    
    try:
        cmd = [sys.executable, str(plots_script)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5min timeout
        
        if result.returncode == 0:
            logger.info("✅ Graphiques générés avec succès")
            return True
        else:
            logger.error(f"❌ Erreur lors de la génération des graphiques: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout lors de la génération des graphiques")
        return False
    except Exception as e:
        logger.error(f"❌ Erreur inattendue: {e}")
        return False


def create_html_report():
    """Crée un rapport HTML récapitulatif"""
    logger.info("📝 Création du rapport HTML...")
    
    benchmarks_dir = Path("../benchmarks")
    html_file = benchmarks_dir / "evaluation_report.html"
    
    # Recherche des fichiers générés
    import glob
    from datetime import datetime
    
    # Trouver les fichiers les plus récents
    result_files = glob.glob(str(benchmarks_dir / "evaluation_results_*.json"))
    plot_files = glob.glob(str(benchmarks_dir / "*.png"))
    
    if not result_files:
        logger.error("❌ Aucun fichier de résultats trouvé")
        return False
    
    latest_results = max(result_files)
    timestamp = Path(latest_results).stem.split('_')[-2] + '_' + Path(latest_results).stem.split('_')[-1]
    
    # Création du HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport d'Évaluation Speech-to-Text</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .meta-info {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .plot-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Rapport d'Évaluation des Modèles Speech-to-Text</h1>
        
        <div class="meta-info">
            <strong>Date de génération:</strong> {datetime.now().strftime("%d/%m/%Y à %H:%M:%S")}<br>
            <strong>Timestamp évaluation:</strong> {timestamp}<br>
            <strong>API utilisée:</strong> http://localhost:8000<br>
            <strong>Dataset:</strong> Common Voice French
        </div>
        
        <h2>📊 Comparaison des Métriques</h2>
        <div class="plot-container">
            <img src="metrics_comparison_{timestamp}.png" alt="Comparaison des métriques">
        </div>
        
        <h2>⚡ Performance vs Précision</h2>
        <div class="plot-container">
            <img src="performance_vs_accuracy_{timestamp}.png" alt="Performance vs Précision">
        </div>
        
        <h2>🕷️ Comparaison Globale (Radar Chart)</h2>
        <div class="plot-container">
            <img src="radar_comparison_{timestamp}.png" alt="Radar Chart">
        </div>
        
        <h2>📋 Tableau Récapitulatif</h2>
        <div class="plot-container">
            <img src="summary_table_{timestamp}.png" alt="Tableau récapitulatif">
        </div>
        
        <h2>📁 Fichiers Générés</h2>
        <ul>
            <li><strong>Résultats JSON:</strong> evaluation_results_{timestamp}.json</li>
            <li><strong>Résultats détaillés:</strong> detailed_results_{timestamp}.json</li>
            <li><strong>Métriques CSV:</strong> metrics_summary_{timestamp}.csv</li>
            <li><strong>Graphiques:</strong> *.png</li>
        </ul>
        
        <div class="footer">
            <p>Généré automatiquement par le système d'évaluation Speech-to-Text</p>
        </div>
    </div>
</body>
</html>
"""
    
    try:
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Rapport HTML créé: {html_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création du rapport HTML: {e}")
        return False


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Évaluation complète des modèles Speech-to-Text")
    parser.add_argument("--samples", "-s", type=int, default=30,
                       help="Nombre d'échantillons à évaluer (défaut: 30)")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Ignorer l'évaluation et générer seulement les graphiques")
    
    args = parser.parse_args()
    
    print("🎯 ÉVALUATION COMPLÈTE DES MODÈLES SPEECH-TO-TEXT")
    print("=" * 50)
    
    start_time = time.time()
    
    # Vérification de l'API
    if not args.skip_evaluation:
        print("🔍 Vérification de l'API...")
        if not check_api_status():
            print("❌ API Speech-to-Text non accessible sur http://localhost:8000")
            print("   Assurez-vous que l'API est démarrée avec:")
            print("   source whisper_env/bin/activate && python api/start_api.py --host 0.0.0.0 --port 8000")
            sys.exit(1)
        print("✅ API accessible")
    
    success = True
    
    # 1. Évaluation des modèles
    if not args.skip_evaluation:
        if not run_evaluation(args.samples):
            success = False
    else:
        print("⏭️  Évaluation ignorée")
    
    # 2. Génération des graphiques
    if success:
        if not generate_plots():
            success = False
    
    # 3. Création du rapport HTML
    if success:
        if not create_html_report():
            success = False
    
    # Résumé final
    total_time = time.time() - start_time
    print("\\n" + "=" * 50)
    
    if success:
        print("🎉 ÉVALUATION TERMINÉE AVEC SUCCÈS !")
        print(f"⏱️  Temps total: {total_time:.1f}s")
        print("📁 Résultats dans: evaluation/benchmarks/")
        print("🌐 Rapport HTML: evaluation/benchmarks/evaluation_report.html")
    else:
        print("❌ ÉVALUATION ÉCHOUÉE")
        print("Consultez les logs ci-dessus pour plus d'informations")
        sys.exit(1)


if __name__ == "__main__":
    main() 