import torch
import logging
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from dotenv import load_dotenv
import os

from connection import get_db_cursor



load_dotenv()
path_dataset = os.getenv("PATH_DATASET")

#CREATE
def create_table_audio():
    with get_db_cursor() as cursor:
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





#READ
def get_audio_path(id_audio, batch):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT path FROM audio WHERE id = %s AND batch = %s", (id_audio, batch))
        return cursor.fetchone()[0]

def get_sentence_originale(id_audio, batch_audio):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT sentence FROM audio WHERE id = %s AND batch = %s", (id_audio, batch_audio))
        return cursor.fetchone()[0]

def get_all_audio():
    with get_db_cursor() as cursor:
        cursor.execute("SELECT id, batch, path, sentence FROM audio")
        return cursor.fetchall()  # Liste de tuples (path, sentence)



def get_number_of_audio():
    with get_db_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM audio")
        return cursor.fetchone()[0]



#UPDATE -> pas d'update


#DELETE
def reset_audio():
    from services.database.results import create_table_results
    with get_db_cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS audio_model_results")
        cursor.execute("DROP TABLE IF EXISTS audio")
    create_table_audio()
    create_table_results()

