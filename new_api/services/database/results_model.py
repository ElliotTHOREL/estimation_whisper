from connection import get_db_cursor

from services.models import get_all_active_models, load_model, unload_model
from services.database.results import translate_many_models_many_audios
from services.database.batch_audio import get_batch_audio_size, get_all_batch_audio

import psutil

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
    mem_before = psutil.virtual_memory().percent
    print(f"Mémoire avant: {mem_before:.1f}%")


    if model not in [nom for (nom, _) in get_all_active_models(app)]:
        load_model(app, model)

    if nom_batch not in [nom for (nom,) in get_all_batch_audio()]:
        raise ValueError(f"Le batch {nom_batch} n'existe pas")

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
        
    print("Résultats ajoutés avec succès")

    mem_before = psutil.virtual_memory().percent
    print(f"Mémoire avant unload: {mem_before:.1f}%")
    unload_model(app, model)
    mem_before = psutil.virtual_memory().percent
    print(f"Mémoire après unload: {mem_before:.1f}%")

def ajoute_result_all_model(app, nom_batch, taille_echantillon, replace):
    for model in get_all_active_models(app):
        ajoute_result_model(app, model, nom_batch, taille_echantillon, replace)


#READ
def get_all_results_model():
    with get_db_cursor() as cursor:
        cursor.execute("SELECT * FROM results_model")
        return cursor.fetchall()

#DELETE
def delete_results_model(id):
    with get_db_cursor() as cursor:
        cursor.execute("DELETE FROM results_model WHERE id = %s", (id,))

def reset_results_model():
    with get_db_cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS results_model")
    create_table_results_model()