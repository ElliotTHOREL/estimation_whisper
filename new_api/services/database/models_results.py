from connection import get_db_cursor

from services.models import load_model, unload_model
from services.database.audio_results import translate_many_models_many_audios
from services.database.batch_audio import get_batch_audio_size, get_all_batch_audio
from services.database.models import get_all_model_names

import psutil
import logging

#CREATE
def create_table_results_model():
    with get_db_cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results_model (
                id INT PRIMARY KEY AUTO_INCREMENT,
                id_model INT,
                nom_batch VARCHAR(100),
                taille_echantillon INT, -- nombre d'audios du batch utilisés
                duree_moyenne FLOAT,
                wer_moyen FLOAT
            )
        """)



def ajoute_result_model(app, model, nom_batch, taille_echantillon, replace):
    if not replace and check_results_model(model, nom_batch, taille_echantillon):
        return
    
    if taille_echantillon > get_batch_audio_size(nom_batch): 
        taille_echantillon = get_batch_audio_size(nom_batch)

    translate_many_models_many_audios(app, [model], nom_batch, 0, taille_echantillon, replace)


    with get_db_cursor() as cursor: 
        cursor.execute("""
            INSERT INTO results_model (id_model, nom_batch, taille_echantillon, duree_moyenne, wer_moyen)
            SELECT 
                id_model,
                batch_audio as nom_batch,
                %s as taille_echantillon,
                AVG(duree) as duree_moyenne,
                AVG(wer) as wer_moyen
            FROM audio_model_results 
            WHERE id_model = (SELECT id FROM modele WHERE name = %s)
            AND batch_audio = %s
            GROUP BY id_model, batch_audio
        """, (taille_echantillon, model, nom_batch))
    
    logging.info(f"Résultats ajoutés pour le modèle {model} ({taille_echantillon} audio du batch {nom_batch})")
        



def generate_all_results(app, nom_batch, taille_echantillon, replace):
    #exclusions = ["seamless-m4t-v2","sb-wav2vec2-fr"]
    exclusions = []

    ma_liste_filtrée = [model for model in get_all_model_names() if model not in exclusions]


    for model in ma_liste_filtrée:
        ajoute_result_model(app, model, nom_batch, taille_echantillon, replace)

#READ
def get_all_results_model():
    with get_db_cursor() as cursor:
        cursor.execute("SELECT * FROM results_model")
        return cursor.fetchall()

def get_results_model(id_model, nom_batch, taille_echantillon):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT duree_moyenne, wer_moyen FROM results_model WHERE id_model = %s AND nom_batch = %s AND taille_echantillon = %s", (id_model, nom_batch, taille_echantillon))
        return cursor.fetchone()

def check_results_model(id_model, nom_batch, taille_echantillon):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM results_model WHERE id_model = %s AND nom_batch = %s AND taille_echantillon = %s", (id_model, nom_batch, taille_echantillon))
        return cursor.fetchone()[0] > 0

#DELETE
def delete_results_model(id):
    with get_db_cursor() as cursor:
        cursor.execute("DELETE FROM results_model WHERE id = %s", (id,))

def reset_results_model():
    with get_db_cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS results_model")
    create_table_results_model()