import torch
import logging
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from dotenv import load_dotenv
import os
import mysql.connector

load_dotenv()
path_dataset = os.getenv("PATH_DATASET")




#CREATE
def load_dataset_interv(deb, fin):
    # Connexion à MariaDB
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()

    # Création de la table avec id manuel
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audio (
        id INT PRIMARY KEY,
        path VARCHAR(255),
        sentence TEXT
    )
    """)
    conn.commit()
    try:
        # Charger le fichier TSV
        path_dataset = os.getenv("PATH_DATASET")
        chemin_tsv = os.path.join(path_dataset, "validated.tsv")

        with open(chemin_tsv, "r", encoding="utf-8") as f:
            next(f)  # sauter l'en-tête
            for i, ligne in enumerate(f):
                if i >= fin:
                    break
                if i >= deb:
                    valeurs = ligne.strip().split("\t")


                    path = valeurs[1]
                    sentence = valeurs[3]

                    cursor.execute("""
                    INSERT IGNORE INTO audio (id, path, sentence)
                    VALUES (%s, %s, %s)
                    """, (i, path, sentence))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def load_dataset(nb_audio):
    load_dataset_interv(0, int(nb_audio))


#READ
def get_audio_path(id_audio):
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
        cursor.execute("SELECT path FROM audio WHERE id = %s", (id_audio,))
        result = cursor.fetchone()[0]
    finally:
        cursor.close()
        conn.close()
    return result

def get_all_audio():
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
        cursor.execute("SELECT id, path, sentence FROM audio")
        result = cursor.fetchall()  # Liste de tuples (path, sentence)
    finally:
        cursor.close()
        conn.close()

    if result is None:
        raise Exception("PROBLEME")
    else:
        return result

def get_number_of_audio():
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
        cursor.execute("SELECT COUNT(*) FROM audio")
        result = cursor.fetchone()[0]   
    finally:
        cursor.close()
        conn.close()

    if result is None:
        raise Exception("PROBLEME")
    else:
        return result

#UPDATE -> pas d'update


#DELETE
def delete_dataset():
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM audio")
        conn.commit()

    except mysql.connector.Error as err:
        if err.errno == 1146:  # Code erreur : table n'existe pas
            print("La table 'audio' n'existe pas.")
    finally:
        cursor.close()
        conn.close()

def delete_dataset_interv(deb, fin):
    #Connexion à MariaDB
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM audio WHERE id BETWEEN %s AND %s", (deb, fin))
        conn.commit()

    except mysql.connector.Error as err:
        if err.errno == 1146:  # Code erreur : table n'existe pas
            print("La table 'audio' n'existe pas.")
    finally:
        cursor.close()
        conn.close()
