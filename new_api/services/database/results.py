import mysql.connector
from services.models import sanity_check_models, get_all_active_models
from services.database.audio import get_all_audio
from services.database.models import ajoute_model

from jiwer import wer


def create_table_results():
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()  
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audio_model_results (
                id_audio INT NOT NULL,
                id_model INT NOT NULL,
                transcription_result TEXT,
                duree FLOAT,
                WER FLOAT,
                PRIMARY KEY (id_audio, id_model),
                FOREIGN KEY (id_audio) REFERENCES audio(id) ON DELETE CASCADE,
                FOREIGN KEY (id_model) REFERENCES modele(id) ON DELETE CASCADE
            )
        """)
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def ajoute_result(model, id_audio, transcription_result, duree, replace):
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()
    try:
        if replace:
            cursor.execute("""
                INSERT INTO audio_model_results (id_audio, id_model, transcription_result, duree)
                VALUES (
                    %s,
                    (SELECT id FROM modele WHERE name = %s),
                    %s,
                    %s
                )
                ON DUPLICATE KEY UPDATE
                    transcription_result = VALUES(transcription_result)
            """, (id_audio, model, transcription_result, duree))
            conn.commit()
        else:
            cursor.execute("""
                INSERT IGNORE INTO audio_model_results (id_audio, id_model, transcription_result, duree)
                VALUES (
                    %s,
                    (SELECT id_model FROM modele WHERE name = %s),
                    %s,
                    %s
                )
            """, (id_audio, model, transcription_result, duree))
            conn.commit()

    
    finally:
        cursor.close()
        conn.close()


def translate_all(app, replace):
    liste_models = [nom for (nom, _) in get_all_active_models(app)]
    translate_many(app, liste_models, replace)

def translate_many(app, liste_models, replace):
    from services.services_translate import translate_one
    sanity_check_models(app)
    for model in liste_models:
        if model not in app.state.models.keys():
            raise ValueError(f"Le modèle {model} n'est pas chargé")
    ajoute_model(liste_models)


    for (id_audio, _ , _) in get_all_audio():
        for model in liste_models:
            transcription_result, durée = translate_one(app, model, id_audio)
            ajoute_result(model, id_audio, transcription_result,durée, replace)


def reset_results():
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()
    
    try:
        cursor.execute("DROP TABLE IF EXISTS audio_model_results")
        conn.commit()
    finally:
        cursor.close()
        conn.close()

    create_table_results()


def estimer_tous_les_wer():
    """calculer le WER pour tous les résultats de la table audio_model_results"""
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()

    try:
        
        cursor.execute("""
            SELECT 
                amr.id_audio,
                amr.id_model,
                amr.transcription_result,
                a.sentence
            FROM audio_model_results amr
            JOIN audio a ON amr.id_audio = a.id
            WHERE amr.transcription_result IS NOT NULL 
            AND a.sentence IS NOT NULL
        """)
        results = cursor.fetchall()
        for result in results:
            id_audio, id_model, transcription_result, sentence = result
            wer_transcription = wer(sentence, transcription_result)
            cursor.execute(
                """
                UPDATE audio_model_results
                SET wer = %s
                WHERE id_audio = %s AND id_model = %s
                """,
                (wer_transcription, id_audio, id_model)
            )
        conn.commit()
    finally:
        cursor.close()
        conn.close()

    return results
