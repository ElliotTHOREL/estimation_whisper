## PRESENTATION GENERALE

 Projet destinée à l'estimation des modèles speech-to-text huggingface.

## UTILISATION RAPIDE

### Mise en place

#### Set up (première utilisation sur une machine)
- cloner le repo : `git clone https://github.com/ElliotTHOREL/estimation_whisper.git`
- créer un venv : `python -m venv venv_ml`
- activer le venv : `source .../estimattion_Whisper/venv_ml/bin/activate`
- télécharger les requirements : `pip install -r requirements.txt`
- créer un `.env` et le compléter accordément à partir du `.env.example`


#### Lancement du projet (A chaque utilisation)
- activer le venv : `source .../estimattion_Whisper/venv_ml/bin/activate`
- créer/allumer la base de données : `docker compose up`
- lancer l'api : `python new_api/main.py`
- lancer le frontend : `streamlit run frontend/main.py`
=> Il est alors possible d'accéder au front ainsi qu'au swagger docs

### Utilisation

#### Ajout de data de test
- Créer un dossier : exemple (nouveaux_audio)
- Mettre tous les nouveaux audios dans un dossier `nouveaux_audio/clips` (le nom **"clips"** est important)
- Créer un fichier **.tsv** de métadonnées et le mettre dans le dossier "nouveaux audios"
  - ce dossier doit avoir une ligne par audio et au moins 2 colonnes 
    - `path` : le nom de l'audio (par exemple pour l'audio  `nouveaux_audio/clips/mon_audio.m4a` -> `mon_audio.m4a`)
    - `sentence` : la transcription réelle de l'audio ou `TDB` si elle n'est pas connue
- Utiliser la route post `/batch_audio_database/load` du swagger docs
  - `nom` = Un nom qui sera utilisée par la suite pour se référer à cet ensemble d'audios
  - `path` = `.../nouveaux_audios/clips`
  - `path_fichier_metadonnees` = `.../nouveaux_audios/_.tsv`
  

#### Ajout de modèles speech-to-text

Dans l'état actuel, le projet garantit son fonctionnement sur 17 "modèles de base" consulatables dans `new_api/services/database/base_models.json`
Il est également possible d'ajouter d'autres modèles...

##### Ajout des modèles de base
- Via le swagger, utiliser la route post `/modeles_database/add_all_base_models`

##### Ajout d'autres modèles
- Via le swagger, utiliser la route post `/modeles_database/add_one`
  - `model` : votre nom du modèle
  - `vrai_modèle` : le nom "hugging_face" du modèle
  - `type_modèle` : (cf : la route get `/modles_database/types_valides`)
    - essayer de rattacher le nouveau modèle au bon "type"
  - `sampling rate` : un paramètre des modèles stt (en général `16 000`)


#### Estimation de modèles

##### Via le swagger
- Consulter les modèles disponibles avec la route get `/modeles_database/all_names`
- Consulter les batchs audio disponibles avec la route get `/batch_audio_database/`
- 2 possibilités pour l'estimation :
  - Estimer **1** modèle: Utiliser la route post `/database_models_results/` avec
    - le nom du modèle à estimer
    - le nom du batch audio
    - le nombre d'audios sur lesquels faire les tests
    - la paramètre `replace` indique le comportement si les audios ont déjà été transcrits par le modèle
      - `false` -> l'ancienne transcription est gardée
      - `true` -> on refait la transcription (utile notamment pour étuider les temps d'exécutions)
  - Estimer **TOUS** les modèles (**long**): Utiliser la route `/database_models_results/all`
    - mêmes paramètres que pour **1** modèle 

##### Via le frontend
- Aller au bas de la page estimations
- Dans le formulaire "Effectuer de nouvelles estimations", choisir :
  - le(s) modèle(s) à tester (si aucun modèle n'st sélectionné, tous les modèles seront estimés)
  - le `batch_audio`
  - le nombre d'audios
- Appuyer sur le bouton submit

#### Visualisation
- Via le frontend
- Aller dans la page visualisation
- Sélectionner les estimations à visualiser (il est possible d'utiliser les filtres)
- Clicker sur "Afficher les graphes"


## STRUCTURE GENERALE

### Composants principaux

- Une API (Fast-API)
- Une Base de données (Maria DB)
- Un Frontend (Streamlit)

### Détails

#### Base de données

La base de données est constituée de 5 tables
- batch_audio
- audio
- modele
- audio_model_result (un résultat par audio et par modèle)
- results_model (un résultat par modèle pour chaque "test" réalisé)

Chaque table dispose de son CRUD dédié avec
- un fichier avec les fonctions de modifs dans new_api/services/database
- un fichier avec les routes correspondantes dans new_api/controllers

La connexion à la BDD est gérée par un système de pooling (cf new_api/connection.py)

#### Logique métier

Toute la logique de transcription est essentiellement gérée dasn 2 fichiers:
 - new_api/services/models.py
 - new_api/services/translate.py

models.py permet d'importer le modèle depuis hugging face jusque la RAM / VRAM
translate.py permet de transcrire un audio avec un modèle importé

Ces 2 fichiers fonctionnent en adaptant le code au "type" du modèle. 
Pour les modèles basiques, il est possible de directement l'utiliser en utilisant une des pipelines existantes (probablement whisper ou wav2vec2)
Pour les modèles plus exotiques, il faut coder les fonctions nécessaires (+ éventullement faire les imports pip et touiller...)
