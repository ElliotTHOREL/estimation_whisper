import csv
from connection import get_db_cursor




#CREATE
def create_table_batch_audio():
    with get_db_cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS batch_audio (
            name VARCHAR(100) PRIMARY KEY,
            path VARCHAR(255) NOT NULL, #path to the folder containing the audio files
            path_fichier_metadonnees VARCHAR(255) NOT NULL #path to the file containing the metadata
        )
        """)


def add_batch_audio(name, path, path_fichier_metadonnees):
    """
    - Add a batch audio to the table batch_audio
    """
    with get_db_cursor() as cursor:
        cursor.execute("INSERT INTO batch_audio (name, path, path_fichier_metadonnees) VALUES (%s, %s, %s)", (name, path, path_fichier_metadonnees))

def add_batch_audio_extended(name, path, path_fichier_metadonnees):
    """
    - Add a batch audio to the table batch_audio
    - Add the audio to the table audio
    """
    add_batch_audio(name, path, path_fichier_metadonnees)

    with get_db_cursor() as cursor:
        with open(path_fichier_metadonnees, newline='', encoding='utf-8') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            
            for i, row in enumerate(reader):
                sentence = row["sentence"]
                path_audio = row["path"]
                cursor.execute("INSERT INTO audio (id, batch, path, sentence) VALUES (%s, %s, %s, %s)", (i, name, path_audio, sentence))



#READ
def get_all_batch_audio():
    with get_db_cursor() as cursor:
        cursor.execute("SELECT name FROM batch_audio")
        return cursor.fetchall()


def get_batch_audio_path(name):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT path FROM batch_audio WHERE name = %s", (name,))
        return cursor.fetchone()[0]

def get_batch_audio_size(name):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM audio WHERE batch = %s", (name,))
        return cursor.fetchone()[0]

#DELETE
def delete_batch_audio(name):
    with get_db_cursor() as cursor:
        cursor.execute("DELETE FROM batch_audio WHERE name = %s", (name,))


def reset_batch_audio():
    from services.database.audio import create_table_audio
    from services.database.results import create_table_results


    with get_db_cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS audio_model_results")
        cursor.execute("DROP TABLE IF EXISTS audio")
        cursor.execute("DROP TABLE IF EXISTS batch_audio")



    create_table_batch_audio()
    create_table_audio()
    create_table_results()