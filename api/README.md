# 🎙️ API Whisper - Modèles de Transcription

Cette API fournit un accès aux modèles Whisper d'OpenAI pour la transcription audio avec mise en cache intelligente.

## 🏗️ Architecture

**L'API contient UNIQUEMENT les modèles Whisper** - les évaluations se font à l'extérieur en appelant cette API.

```
📁 Projet/
├── 📁 api/                    # ⚡ API des modèles (SEULEMENT)
│   ├── whisper_api.py         # Serveur FastAPI avec cache
│   ├── start_api.py           # Script de démarrage
│   └── README.md              # Documentation API
├── evaluate_with_api.py       # 🔬 Évaluations externes
├── evaluation_with_api.py     # Fonctions d'évaluation
├── whisper_client.py          # Client pour appeler l'API
├── evaluation/code/metrics.py # Calcul des métriques
└── plot_eval_results.py       # Génération des graphiques
```

## 🚀 Démarrage rapide

### 1. Démarrer l'API

```bash
# Méthode 1: Script de démarrage
cd api
python start_api.py --host 0.0.0.0 --port 8000

# Méthode 2: Directement
python api/whisper_api.py --host 0.0.0.0 --port 8000 --reload
```

### 2. Tester l'API

```bash
# Vérifier la santé
curl http://localhost:8000/health

# Lister les modèles disponibles
curl http://localhost:8000/models
```

### 3. Faire une évaluation (depuis la racine)

```bash
# Évaluation complète avec démarrage automatique de l'API
python evaluate_with_api.py --clips 21-25

# Avec API déjà démarrée
python evaluate_with_api.py --clips 1-10 --api-url http://localhost:8000
```

## 📡 Endpoints de l'API

### 🎯 Transcription

#### `POST /transcribe`
Transcrit un fichier audio uploadé.

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "audio_file=@audio.mp3" \
  -F "model_size=base" \
  -F "language=fr"
```

#### `POST /transcribe_file`  
Transcrit un fichier via son chemin sur le serveur.

```bash
curl -X POST "http://localhost:8000/transcribe_file" \
  -F "file_path=/path/to/audio.mp3" \
  -F "model_size=large-v3" \
  -F "language=fr"
```

### 📊 Gestion

#### `GET /models`
Liste des modèles disponibles.

```json
{
  "available_models": ["tiny", "base", "small", "medium", "large", "large-v3"],
  "model_details": {
    "tiny": {"size": "~39 MB", "speed": "~32x"},
    "base": {"size": "~74 MB", "speed": "~16x"},
    "small": {"size": "~244 MB", "speed": "~6x"},
    "medium": {"size": "~769 MB", "speed": "~2x"},
    "large": {"size": "~1550 MB", "speed": "~1x"},
    "large-v3": {"size": "~1550 MB", "speed": "~1x"}
  }
}
```

#### `GET /cache/info`
Informations sur le cache des modèles.

#### `POST /cache/clear`
Vide le cache des modèles.

#### `GET /health`
État de santé de l'API.

## ⚡ Mise en cache intelligente

L'API maintient un cache LRU (Least Recently Used) des modèles :
- **Maximum 3 modèles** en mémoire simultanément
- **Éviction automatique** des modèles les moins utilisés
- **Accès rapide** aux modèles populaires

## 🔧 Configuration

### Variables d'environnement

```bash
export LANGUE=fr                    # Langue par défaut
export WHISPER_CACHE_DIR=/tmp/cache # Dossier cache (optionnel)
```

### Paramètres de démarrage

```bash
python api/start_api.py \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --reload
```

## 📈 Utilisation pour les évaluations

L'API est conçue pour être appelée par des scripts d'évaluation externes :

1. **Démarrage** : L'API démarre et charge les modèles à la demande
2. **Évaluation** : Les scripts externes appellent l'API pour chaque transcription
3. **Cache** : Les modèles restent en mémoire pour les appels suivants
4. **Métriques** : Calculées côté client, pas dans l'API

### Exemple d'usage programmatique

```python
from whisper_client import WhisperAPIClient

# Connexion à l'API
client = WhisperAPIClient("http://localhost:8000")

# Transcription
result = client.transcribe_file(
    file_path="audio.mp3",
    model_size="large-v3",
    language="fr"
)

print(result["text"])
print(f"Temps d'inférence: {result['inference_time']:.2f}s")
```

## 📦 Dépendances

```bash
# Installation (dans l'environnement virtuel)
pip install fastapi uvicorn transformers torch librosa
```

## 🐛 Debugging

### Logs de l'API
```bash
# Démarrage avec logs détaillés
python api/start_api.py --log-level debug
```

### Vérification du cache
```bash
curl http://localhost:8000/cache/info
```

### Test de connectivité
```bash
curl http://localhost:8000/health
```

## 🚨 Notes importantes

- **CPU vs GPU** : L'API détecte automatiquement CUDA
- **Mémoire** : Surveiller l'usage mémoire avec plusieurs modèles
- **Concurrence** : L'API gère les requêtes simultanées
- **Sécurité** : En production, configurer HTTPS et authentification 