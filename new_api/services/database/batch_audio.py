import mysql.connector
import csv




#CREATE
def create_table_batch_audio():
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
        CREATE TABLE IF NOT EXISTS batch_audio (
            name VARCHAR(100) PRIMARY KEY,
            path VARCHAR(255) NOT NULL, #path to the folder containing the audio files
            path_fichier_metadonnees VARCHAR(255) NOT NULL #path to the file containing the metadata
        )
        """)
        conn.commit()
    finally:
        cursor.close()
        conn.close()    

def add_batch_audio(name, path, path_fichier_metadonnees):
    """
    - Add a batch audio to the table batch_audio
    """
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO batch_audio (name, path, path_fichier_metadonnees) VALUES (%s, %s, %s)", (name, path, path_fichier_metadonnees))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def add_batch_audio_extended(name, path, path_fichier_metadonnees):
    """
    - Add a batch audio to the table batch_audio
    - Add the audio to the table audio
    """
    add_batch_audio(name, path, path_fichier_metadonnees)

    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()
    try:
        with open(path_fichier_metadonnees, newline='', encoding='utf-8') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            
            for i, row in enumerate(reader):
                sentence = row["sentence"]
                path_audio = row["path"]
                cursor.execute("INSERT INTO audio (id, batch, path, sentence) VALUES (%s, %s, %s, %s)", (i, name, path_audio, sentence))
            conn.commit()

    finally:
        cursor.close()
        conn.close()



    

#READ
def get_all_batch_audio():
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()

    result = None
    try:
        cursor.execute("SELECT name FROM batch_audio")
        result = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()
    return result




#DELETE
def delete_batch_audio(name):
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM batch_audio WHERE name = %s", (name,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def reset_batch_audio():
    from services.database.audio import create_table_audio
    from services.database.results import create_table_results


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
        cursor.execute("DROP TABLE IF EXISTS audio")
        cursor.execute("DROP TABLE IF EXISTS batch_audio")
        conn.commit()
    finally:
        cursor.close()
        conn.close()


    create_table_batch_audio()
    create_table_audio()
    create_table_results()