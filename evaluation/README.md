# 📊 Système d'Évaluation des Modèles Whisper

Ce dossier contient un système complet d'évaluation des modèles Whisper utilisant l'API pour des comparaisons de performance et de qualité.

## 📁 Structure

```
evaluation/
├── code/                           # Scripts d'évaluation
│   ├── evaluate_whisper_models.py  # Script principal d'évaluation
│   ├── generate_plots.py           # Génération des graphiques
│   ├── detailed_error_analysis.py  # Analyse détaillée des erreurs
│   └── run_full_evaluation.py      # Script complet automatisé
├── benchmarks/                     # Résultats et graphiques
│   ├── evaluation_results_*.json   # Résultats JSON
│   ├── detailed_results_*.json     # Résultats détaillés
│   ├── metrics_summary_*.csv       # Métriques au format CSV
│   ├── *.png                       # Graphiques générés
│   └── evaluation_report.html      # Rapport HTML
└── README.md                       # Ce fichier
```

## 🚀 Utilisation Rapide

### Prérequis
1. **API Whisper** doit être lancée :
   ```bash
   source whisper_env/bin/activate
   python api/start_api.py --host 0.0.0.0 --port 8000
   ```

2. **Modèles chargés** dans l'API (recommandé pour des résultats rapides)

### Évaluation complète automatisée
```bash
# Évaluation sur 30 échantillons (recommandé)
python evaluation/code/run_full_evaluation.py --samples 30

# Évaluation rapide sur 10 échantillons
python evaluation/code/run_full_evaluation.py --samples 10

# Regénérer seulement les graphiques
python evaluation/code/run_full_evaluation.py --skip-evaluation
```

## 🔧 Scripts Individuels

### 1. Évaluation des Modèles
```bash
# Script d'évaluation principal
python evaluation/code/evaluate_whisper_models.py
```

**Ce qu'il fait :**
- Utilise l'API Whisper pour transcrire des échantillons audio
- Calcule les métriques WER, CER, BLEU, MER
- Mesure les temps d'inférence
- Sauvegarde les résultats en JSON et CSV

### 2. Génération des Graphiques
```bash
# Génère tous les graphiques
python evaluation/code/generate_plots.py

# Utiliser un fichier de résultats spécifique
python evaluation/code/generate_plots.py --results-file evaluation/benchmarks/evaluation_results_20250623_152800.json
```

**Graphiques générés :**
- **Comparaison des métriques** : Barres comparatives pour WER, CER, BLEU, MER
- **Performance vs Précision** : Scatter plots vitesse vs qualité
- **Radar Chart** : Vue globale de tous les modèles
- **Tableau récapitulatif** : Résumé formaté

### 3. Analyse Détaillée des Erreurs
```bash
# Analyse détaillée des erreurs (substitutions, insertions, suppressions)
python evaluation/code/detailed_error_analysis.py

# Utiliser un fichier de résultats détaillés spécifique
python evaluation/code/detailed_error_analysis.py --results-file evaluation/benchmarks/detailed_results_20250623_152800.json
```

**Analyses générées :**
- **Rapports détaillés par modèle** : WER par clip avec décomposition des erreurs
- **Graphiques de distribution des erreurs** : Répartition par type d'erreur
- **Histogrammes WER** : Distribution des performances par clip
- **Exemples des pires erreurs** : Cas d'échec avec analyse

## 📈 Métriques Calculées

| Métrique | Description | Interprétation |
|----------|-------------|----------------|
| **WER** | Word Error Rate | Plus bas = meilleur |
| **CER** | Character Error Rate | Plus bas = meilleur |
| **BLEU** | BiLingual Evaluation Understudy | Plus haut = meilleur |
| **MER** | Match Error Rate | Plus bas = meilleur |
| **Temps d'inférence** | Temps moyen par audio | Plus bas = plus rapide |
| **Taux de succès** | % de transcriptions réussies | Plus haut = plus fiable |

## 🎯 Exemples d'Utilisation

### Évaluation Standard
```bash
# Évaluation complète sur 30 échantillons
python evaluation/code/run_full_evaluation.py --samples 30
```

### Évaluation Rapide pour Tests
```bash
# Test rapide sur 5 échantillons
python evaluation/code/run_full_evaluation.py --samples 5
```

### Analyse d'Anciens Résultats
```bash
# Regénérer graphiques à partir de résultats existants
python evaluation/code/generate_plots.py --results-file evaluation/benchmarks/evaluation_results_20250623_152800.json
```

## 📊 Interprétation des Résultats

### Graphique Comparaison des Métriques
- **WER/CER/MER** : Plus la barre est basse, meilleur est le modèle
- **BLEU** : Plus la barre est haute, meilleur est le modèle

### Graphique Performance vs Précision
- **Coin bas-gauche** : Rapide ET précis (idéal)
- **Coin bas-droite** : Lent mais précis
- **Coin haut-gauche** : Rapide mais imprécis

### Radar Chart
- **Périmètre large** : Modèle polyvalent
- **Forme équilibrée** : Performance consistante
- **Pics** : Points forts spécifiques

## 🔧 Configuration

### Modification du Nombre d'Échantillons
Éditez `evaluation/code/evaluate_whisper_models.py` ligne ~202 :
```python
results = evaluator.run_evaluation(num_samples=50)  # Modifier ici
```

### Ajout de Nouveaux Modèles
Éditez `evaluation/code/evaluate_whisper_models.py` ligne ~43 :
```python
self.models = ["tiny", "base", "small", "medium", "large", "large-v3", "nouveau_modele"]
```

### Personnalisation des Graphiques
Modifiez `evaluation/code/generate_plots.py` :
- Couleurs : ligne ~38
- Tailles : ligne ~36
- Métriques : lignes ~58, ~168

## 📁 Fichiers de Sortie

### Résultats JSON
```json
{
  "tiny": {
    "wer": 0.245,
    "cer": 0.123,
    "bleu": 0.678,
    "mer": 0.189,
    "avg_inference_time": 0.45,
    "success_rate": 98.5
  }
}
```

### CSV Métriques
| Model | WER | CER | BLEU | MER | Temps | Succès |
|-------|-----|-----|------|-----|-------|--------|
| tiny  | 0.245 | 0.123 | 0.678 | 0.189 | 0.45s | 98.5% |

## 🚨 Dépannage

### API Non Accessible
```bash
# Vérifier l'état de l'API
curl http://localhost:8000/health

# Redémarrer l'API
source whisper_env/bin/activate
python api/start_api.py --host 0.0.0.0 --port 8000
```

### Erreurs de Mémoire
- Réduire le nombre d'échantillons avec `--samples 10`
- Vider le cache API : `curl -X POST http://localhost:8000/cache/clear`

### Graphiques Non Générés
```bash
# Vérifier les dépendances
pip install matplotlib seaborn pandas

# Test de génération manuelle
python evaluation/code/generate_plots.py
```

## 📞 Support

Pour des questions spécifiques :
1. Consultez les logs détaillés des scripts
2. Vérifiez que l'API fonctionne correctement
3. Validez que le dataset Common Voice est accessible

---

## 🎉 Résultats Attendus

Après une évaluation complète, vous obtiendrez :
- ✅ Comparaison objective de tous les modèles
- ✅ Graphiques publication-ready
- ✅ Rapport HTML navigable
- ✅ Données exportables (CSV, JSON)
- ✅ Métriques de performance détaillées 