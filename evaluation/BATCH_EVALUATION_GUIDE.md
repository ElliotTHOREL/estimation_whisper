# 📊 Guide d'Évaluation par Batch

Ce guide vous explique comment évaluer vos modèles par batch pour optimiser l'utilisation de la RAM.

## 🎯 Vue d'Ensemble

L'évaluation par batch résout le problème de RAM insuffisante en :
- 🔄 Divisant les modèles en groupes (batches)
- 🧹 Vidant le cache entre les batches
- 📊 Consolidant les résultats finaux
- 🎨 Générant les graphiques de comparaison

## 📁 Nouveaux Scripts Disponibles

| Script | Description | Usage Principal |
|--------|-------------|-----------------|
| `batch_evaluation.py` | Évaluation par batch manuelle | Contrôle précis des batches |
| `memory_monitor.py` | Surveillance RAM/GPU | Monitoring pendant l'évaluation |
| `resource_manager.py` | Gestionnaire automatique | Analyse et évaluation optimisée |

## 🚀 Utilisation Rapide

### 1. Mode Automatique (Recommandé)
```bash
# Analyse automatique + évaluation optimisée
cd evaluation/code
python resource_manager.py --auto --samples 30
```

### 2. Analyse des Ressources Seulement
```bash
# Voir les recommandations sans lancer l'évaluation
python resource_manager.py --analyze-only
```

### 3. Mode Interactif
```bash
# Menu interactif avec options
python resource_manager.py
```

## 📋 Stratégies d'Évaluation

### 🏃‍♂️ Stratégie Agressive (RAM > 15GB)
- **Batches larges** : 4-5 modèles par batch
- **Temps** : ~1h pour tous les modèles
- **Exemples** : Whisper large + plusieurs Wav2Vec2

### ⚖️ Stratégie Modérée (8-15GB RAM)
- **Batches moyens** : 3 modèles par batch
- **Temps** : ~1h30 pour tous les modèles
- **Équilibrage** : Modèles lourds séparés

### 🐌 Stratégie Conservative (< 8GB RAM)
- **Petits batches** : 2 modèles maximum
- **Temps** : ~2h pour tous les modèles
- **Sécurité** : Évite les dépassements mémoire

## 🛠️ Utilisation Avancée

### Évaluation de Batches Spécifiques
```bash
# Évaluer seulement les modèles Whisper légers
python batch_evaluation.py --batches batch_1_light --samples 20

# Évaluer plusieurs batches
python batch_evaluation.py --batches batch_1_light batch_2_medium --samples 30
```

### Monitoring en Parallèle
```bash
# Terminal 1 : Monitoring
python memory_monitor.py --threshold 85 --log-interval 30

# Terminal 2 : Évaluation
python batch_evaluation.py --samples 30
```

### Lister les Batches Disponibles
```bash
python batch_evaluation.py --list-batches
```

## 📊 Organisation des Batches

### Batch 1 - Modèles Légers (~4GB)
```
- whisper:tiny      (1GB)
- whisper:base      (1GB)
- whisper:small     (2GB)
```

### Batch 2 - Modèles Moyens (~10.5GB)
```
- whisper:medium    (5GB)
- wav2vec2:xlsr-53-french (3GB)
- wav2vec2:bhuang-french  (2.5GB)
```

### Batch 3 - Modèles Lourds (~20GB)
```
- whisper:large     (10GB)
- whisper:large-v3  (10GB)
```

### Batch 4 - Wav2Vec2 Spécialisés (~9GB)
```
- wav2vec2:jonatasgrosman-xlsr      (3.5GB)
- wav2vec2:jonatasgrosman-voxpopuli (3GB)
- wav2vec2:wasertech-cv9            (2.5GB)
```

### Batch 5 - Modèles Seamless (~18GB)
```
- seamless:medium   (6GB)
- seamless:large    (12GB)
```

### Batch 6 - Modèles Gemma (~23GB)
```
- gemma3n:e2b      (8GB)
- gemma3n:e4b      (15GB)
```

## 🔍 Monitoring et Optimisation

### Métriques Surveillées
- **RAM** : Utilisation en pourcentage et GB
- **GPU** : Mémoire allouée/totale (si CUDA disponible)
- **Swap** : Utilisation du fichier d'échange
- **Seuils** : Alertes automatiques (défaut: 85%)

### Fichiers de Monitoring
```
evaluation/benchmarks/
├── memory_stats_TIMESTAMP.json     # Statistiques détaillées
├── batch_X_results_TIMESTAMP.json  # Résultats par batch
├── consolidated_results_TIMESTAMP.json # Résultats finaux
└── *.png                           # Graphiques générés
```

## ⚡ Optimisations

### Réduction de la Consommation Mémoire
1. **Réduire les échantillons** : `--samples 10` au lieu de 30
2. **Vider le cache** : Automatique entre les batches
3. **Modèles sélectifs** : Évaluer seulement les modèles critiques

### Accélération
1. **GPU VRAM** : Décharge automatique sur GPU si disponible
2. **Batches parallèles** : Possible si RAM très importante (>32GB)
3. **Cache intelligent** : Réutilisation des transcriptions communes

## 🚨 Gestion des Erreurs

### RAM Insuffisante
```bash
# Solution 1 : Réduire la taille des batches
python batch_evaluation.py --batches batch_1_light --samples 10

# Solution 2 : Mode ultra-conservateur
python resource_manager.py  # Choisir option 4 (test rapide)
```

### API Non Accessible
```bash
# Vérifier l'API
curl http://localhost:8000/health

# Redémarrer l'API si nécessaire
cd ../../api
python start_api.py --host 0.0.0.0 --port 8000
```

### Timeout ou Crash
```bash
# Reprendre où on s'est arrêté
python batch_evaluation.py --batches batch_3_heavy batch_4_wav2vec --samples 30
```

## 📈 Interprétation des Résultats

### Fichiers Générés
- **`consolidated_results_*.json`** : Métriques de tous les modèles
- **`consolidated_detailed_*.json`** : Résultats détaillés par échantillon
- **Graphiques** : Comparaisons automatiques (WER, BLEU, vitesse)

### Métriques Clés
- **WER** : Plus bas = meilleur (erreurs de mots)
- **BLEU** : Plus haut = meilleur (qualité globale)
- **Temps** : Plus bas = plus rapide
- **Succès** : Plus haut = plus fiable

## 🎯 Workflow Recommandé

1. **Analyse initiale**
   ```bash
   python resource_manager.py --analyze-only
   ```

2. **Test rapide** (pour vérifier que tout fonctionne)
   ```bash
   python resource_manager.py --auto --samples 5
   ```

3. **Évaluation complète**
   ```bash
   python resource_manager.py --auto --samples 30
   ```

4. **Analyse des résultats**
   - Consulter les graphiques générés
   - Examiner `consolidated_results_*.json`
   - Vérifier les statistiques de mémoire

## 💡 Conseils Pratiques

### Pour Systèmes avec RAM Limitée (< 8GB)
- Utiliser `--samples 10` maximum
- Évaluer 1-2 modèles à la fois
- Surveiller le swap activement

### Pour Systèmes Performants (> 16GB)
- Utiliser `--samples 50` pour plus de précision
- Batches plus larges possibles
- Monitoring moins critique

### Pour Tests de Développement
- Utiliser `--samples 5` pour des tests rapides
- Se concentrer sur 2-3 modèles représentatifs
- Itérer rapidement

## 🔧 Personnalisation

### Modifier les Estimations Mémoire
Éditez `resource_manager.py` ligne ~60 :
```python
def get_model_memory_estimates(self) -> Dict[str, float]:
    return {
        "whisper:tiny": 1.0,    # Ajustez selon vos observations
        "whisper:base": 1.0,
        # ... autres modèles
    }
```

### Ajuster les Seuils
```bash
# Seuil d'alerte RAM plus strict
python memory_monitor.py --threshold 75

# Intervalle de monitoring plus fréquent
python memory_monitor.py --log-interval 15
```

### Créer des Batches Personnalisés
Modifiez `batch_evaluation.py` ligne ~40 pour définir vos propres groupes de modèles.

---

Ce système vous permet d'évaluer tous vos modèles de manière optimisée, même avec des ressources limitées ! 🚀 