import mysql.connector
from services.models import  get_all_active_models
from services.database.audio import get_all_audio, get_sentence_originale

from jiwer import wer
from connection import get_db_cursor




#CREATE
def create_table_results():
    with get_db_cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audio_model_results (
                id_audio INT NOT NULL,
                batch_audio VARCHAR(100) NOT NULL,
                id_model INT NOT NULL,
                transcription_result TEXT,
                duree FLOAT,
                WER FLOAT,
                FOREIGN KEY (id_audio) REFERENCES audio(id) ON DELETE CASCADE,
                FOREIGN KEY (batch_audio) REFERENCES batch_audio(name) ON DELETE CASCADE,
                FOREIGN KEY (id_model) REFERENCES modele(id) ON DELETE CASCADE
            )
        """)



def ajoute_result(model, id_audio, batch_audio, transcription_result, duree, replace):
    with get_db_cursor() as cursor:
        sentence_originale = get_sentence_originale(id_audio, batch_audio)
        wer_transcription = wer(sentence_originale, transcription_result)

        if replace:
            # Supprime l'enregistrement existant s'il existe
            cursor.execute("""
                DELETE FROM audio_model_results 
                WHERE id_audio = %s 
                AND batch_audio = %s 
                AND id_model = (SELECT id FROM modele WHERE name = %s)
            """, (id_audio, batch_audio, model))
            
            # Ajoute le nouvel enregistrement
            cursor.execute("""
                INSERT INTO audio_model_results (id_audio, batch_audio, id_model, transcription_result, duree, wer)
                VALUES (
                    %s,
                    %s,
                    (SELECT id FROM modele WHERE name = %s),
                    %s,
                    %s,
                    %s
                )
            """, (id_audio, batch_audio, model, transcription_result, duree, wer_transcription))

        else:
            # Vérifie si le triplet existe déjà
            cursor.execute("""
                SELECT COUNT(*) FROM audio_model_results 
                WHERE id_audio = %s 
                AND batch_audio = %s 
                AND id_model = (SELECT id FROM modele WHERE name = %s)
            """, (id_audio, batch_audio, model))
            
            exists = cursor.fetchone()[0] > 0
            
            # N'ajoute que si le triplet n'existe pas
            if not exists:
                cursor.execute("""
                    INSERT INTO audio_model_results (id_audio, batch_audio, id_model, transcription_result, duree, wer)
                    VALUES (
                        %s,
                        %s,
                        (SELECT id FROM modele WHERE name = %s),
                        %s,
                        %s,
                        %s
                    )
                """, (id_audio, batch_audio, model, transcription_result, duree, wer_transcription))



def translate_many_models_many_audios(app, liste_models, batch_audio, deb, fin, replace ):
    from services.translate import translate_one

    for id_audio in range(deb, fin):
        for model in liste_models:
            if replace or not check_results(id_audio, batch_audio, model):
                transcription_result, durée = translate_one(app, model, id_audio, batch_audio)
                ajoute_result(model, id_audio, batch_audio, transcription_result,durée, replace)


def translate_all_models_many_audios(app, replace, batch_audio, deb, fin):
    liste_models = [nom for (nom, _) in get_all_active_models(app)]
    translate_many_models_many_audios(app, liste_models, batch_audio, deb, fin, replace) 
        

def translate_many_models_all_audios(app, liste_models, replace):
    from services.translate import translate_one

    for (id_audio, batch_audio, _ , _) in get_all_audio():
        for model in liste_models:
            if replace or not check_results(id_audio, batch_audio, model):
                transcription_result, durée = translate_one(app, model, id_audio, batch_audio)
                ajoute_result(model, id_audio, batch_audio, transcription_result,durée, replace)


def translate_all_models_all_audios(app, replace):
    liste_models = [nom for (nom, _) in get_all_active_models(app)]
    translate_many_models_all_audios(app, liste_models, replace)

#READ
def check_results(id_audio, nom_batch, nom_model):
    with get_db_cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(*) 
            FROM audio_model_results amr
            JOIN modele m ON amr.id_model = m.id
            WHERE amr.id_audio = %s 
            AND amr.batch_audio = %s 
            AND m.name = %s
        """, (id_audio, nom_batch, nom_model))
        result = cursor.fetchone()[0]
        return result > 0

#DELETE
def reset_results():
    with get_db_cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS audio_model_results")
    create_table_results()


def estimer_tous_les_wer():
    """calculer le WER pour tous les résultats de la table audio_model_results"""
    with get_db_cursor() as cursor:
        
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


    return results
