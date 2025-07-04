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
def create_table_audio():
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
        CREATE TABLE IF NOT EXISTS audio (
            id INT,
            batch VARCHAR(100),
            path VARCHAR(255),
            sentence TEXT,
            FOREIGN KEY (batch) REFERENCES batch_audio(name) ON DELETE CASCADE,
            PRIMARY KEY (id, batch)
        )
        """)
        conn.commit()
    finally:
        cursor.close()
        conn.close()




#READ
def get_audio_path(id_audio, batch):
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
        cursor.execute("SELECT path FROM audio WHERE id = %s AND batch = %s", (id_audio, batch))
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
        cursor.execute("SELECT id, batch, path, sentence FROM audio")
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
def reset_dataset():
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
        conn.commit()
    finally:
        cursor.close()
        conn.close()

    create_table_audio()
    create_table_results()

