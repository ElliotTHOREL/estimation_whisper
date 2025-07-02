# 🎙️ Système d'Évaluation Whisper avec API

Système complet d'évaluation des modèles Whisper d'OpenAI avec architecture microservices séparée.

## 🏗️ Architecture

**Principe clé** : Séparation complète entre les modèles et les évaluations

```
📁 Projet Whisper/
├── 🔧 API (modèles uniquement)
│   ├── whisper_api.py      # Serveur FastAPI avec cache
│   ├── start_api.py        # Script de démarrage
│   └── README.md           # Documentation API
├── 🔬 Évaluations (externes)
│   ├── evaluate_with_api.py     # Script principal d'évaluation
│   ├── evaluation_with_api.py   # Fonctions d'évaluation
│   ├── whisper_client.py        # Client HTTP pour l'API
│   │   ├── metrics.py           # Calcul des métriques
│   └── plot_eval_results.py     # Génération graphiques
├── 📊 Données
│   └── cv-corpus-21.0-delta-2025-03-14/fr/
└── 📈 Résultats
    ├── eval_results_20clips/
    ├── eval_results_21_25/
    └── eval_results_X_Y_api/
```

## 🚀 Démarrage rapide

### 1. Installation

```bash
# Activation de l'environnement
source whisper_env/bin/activate

# Vérification des dépendances
pip install -r requirements.txt
```

### 2. Test de l'architecture

```bash
# Test complet de l'architecture séparée
python test_api_architecture.py
```

### 3. Évaluation complète

```bash
# Évaluation avec démarrage automatique de l'API
python evaluate_with_api.py --clips 21-25

# Avec options personnalisées
python evaluate_with_api.py \
  --clips 1-10 \
  --models tiny base large-v3 \
  --api-url http://localhost:8000 \
  --output-dir my_eval_results
```

## 🎯 Utilisation

### Option 1: Évaluation automatique (recommandé)

```bash
# Le script gère tout automatiquement
python evaluate_with_api.py --clips 21-25
```

**Ce qui se passe :**
1. ✅ Vérifie si l'API est déjà en cours
2. 🚀 Démarre l'API si nécessaire
3. 🔬 Évalue tous les modèles
4. 📈 Génère les graphiques
5. 🛑 Arrête l'API à la fin

### Option 2: API manuelle + évaluations

```bash
# Terminal 1: Démarrer l'API
cd api
python start_api.py --host 0.0.0.0 --port 8000

# Terminal 2: Lancer les évaluations
python evaluate_with_api.py --clips 21-25 --api-url http://localhost:8000
```

### Option 3: Évaluation locale (sans API)

```bash
# Méthode traditionnelle sans API
python example_evaluation.py --start 21 --end 25
```

## 📡 API Whisper

### Endpoints principaux

- `POST /transcribe_file` - Transcription d'un fichier
- `GET /models` - Liste des modèles disponibles
- `GET /cache/info` - État du cache
- `GET /health` - Santé de l'API

### Cache intelligent

- **3 modèles maximum** en mémoire simultanément
- **Éviction LRU** automatique
- **Performance optimisée** pour les évaluations répétées

### Documentation complète

```bash
# Démarrer l'API
python api/start_api.py

# Accéder à la doc interactive
open http://localhost:8000/docs
```

## 🔬 Évaluations

### Métriques calculées

- **WER** (Word Error Rate)
- **CER** (Character Error Rate)  
- **MER** (Match Error Rate)
- **WIL** (Word Information Lost)
- **BLEU** (Bilingual Evaluation Understudy)
- **ROUGE-L** (Recall-Oriented Understudy for Gisting Evaluation)

### Modèles supportés

- `tiny` (~39 MB, ~32x vitesse)
- `base` (~74 MB, ~16x vitesse)
- `small` (~244 MB, ~6x vitesse)
- `medium` (~769 MB, ~2x vitesse)
- `large` (~1550 MB, ~1x vitesse)
- `large-v3` (~1550 MB, ~1x vitesse, dernière version)

### Formats de sortie

- **JSON** : Prédictions et métriques détaillées
- **CSV** : Comparaison tabulaire des modèles
- **PNG** : Graphiques barplots par métrique

## 🔧 Configuration

### Variables d'environnement

```bash
export LANGUE=fr                    # Langue par défaut
export WHISPER_CACHE_DIR=/tmp/cache # Dossier cache modèles
```

### Paramètres d'évaluation

```bash
python evaluate_with_api.py \
  --clips 1-50 \                    # Range de clips à tester
  --models tiny base large-v3 \     # Modèles à évaluer
  --language fr \                   # Langue de transcription
  --api-url http://localhost:8000 \ # URL de l'API
  --output-dir results_custom \     # Dossier de sortie
  --keep-api                        # Garder l'API en cours
```

## 📊 Exemples de résultats

### Évaluation sur 5 clips (clips 21-25)

| Modèle    | WER   | CER   | BLEU  | Temps moyen |
|-----------|-------|-------|-------|-------------|
| large-v3  | 0.245 | 0.089 | 0.756 | 1.8s        |
| large     | 0.267 | 0.098 | 0.728 | 1.9s        |
| medium    | 0.289 | 0.112 | 0.698 | 0.8s        |
| base      | 0.334 | 0.145 | 0.645 | 0.4s        |
| tiny      | 0.456 | 0.203 | 0.534 | 0.2s        |

### Graphiques générés

- `barplot_avg_wer.png` - Comparaison WER
- `barplot_avg_cer.png` - Comparaison CER  
- `barplot_avg_bleu.png` - Comparaison BLEU
- (et 5 autres métriques)

## 🚨 Avantages de l'architecture API

### ✅ Avant (problèmes)

- Rechargement modèles à chaque évaluation
- Consommation mémoire excessive
- Pas de réutilisation entre évaluations
- Couplage fort modèles/évaluations

### ✅ Après (solutions)

- **Cache intelligent** : Modèles restent en mémoire
- **Réutilisation** : Un modèle sert plusieurs évaluations
- **Séparation** : API modèles ↔ Scripts évaluations
- **Scalabilité** : API peut servir plusieurs clients

### Performance

```bash
# Sans API: 6 modèles × 5 clips = 30 chargements
# Avec API: 6 modèles × 1 chargement = 6 chargements (cache)
# Gain: ~80% de temps de chargement
```

## 🔍 Debugging

### Vérification API

```bash
# API disponible ?
curl http://localhost:8000/health

# Modèles en cache ?
curl http://localhost:8000/cache/info

# Test transcription
curl -X POST http://localhost:8000/transcribe_file \
  -F "file_path=cv-corpus-21.0-delta-2025-03-14/fr/clips/common_voice_fr_41911225.mp3" \
  -F "model_size=tiny"
```

### Logs détaillés

```bash
# API avec logs debug
python api/start_api.py --log-level debug

# Évaluation avec traces
python evaluate_with_api.py --clips 21-22 --models tiny
```

## 🤝 Contribuer

Le système est modulaire et extensible :

- **Nouvelles métriques** : Ajouter dans `evaluation/code/metrics.py`
- **Nouveaux modèles** : Modifier `whisper_models.py`
- **Nouvelles visualisations** : Étendre `plot_eval_results.py`
- **Nouveaux endpoints** : Ajouter dans `api/whisper_api.py`

## 📝 Licence

Projet d'évaluation des modèles Whisper pour la recherche académique. 