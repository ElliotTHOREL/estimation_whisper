from connection import get_db_cursor

from enum import Enum
from pydantic import BaseModel
import json
import logging

TYPES_VALIDES = {"whisper", "wav2vec2", "kyutai", "speechbrain_seq2seq", "speechbrain_ctc", "seamless"}

#CREATE
def create_table_models():
    with get_db_cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS modele (
                id INT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(100) NOT NULL UNIQUE,
                vrai_modele TEXT NOT NULL,
                type_modele VARCHAR(100) NOT NULL,
                sampling_rate INT NOT NULL
            )
        """)

#READ
def get_all_model_names():
    with get_db_cursor() as cursor:
        cursor.execute("SELECT name FROM modele")
        return [name for (name,) in cursor.fetchall()]
    
def get_all_models():
    with get_db_cursor() as cursor:
        cursor.execute("SELECT * FROM modele")
        return cursor.fetchall()

def get_types_valides():
    return TYPES_VALIDES


def find_type_modele(name_model):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT type_modele FROM modele WHERE name = %s", (name_model,))
        return cursor.fetchone()[0]

def find_vrai_modele(name_model):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT vrai_modele FROM modele WHERE name = %s", (name_model,))
        return cursor.fetchone()[0]

def find_sr_modele(name_model):
    with get_db_cursor() as cursor:
        cursor.execute("SELECT sampling_rate FROM modele WHERE name = %s", (name_model,))
        return cursor.fetchone()[0]

#UPDATE
def ajoute_model(model:str, vrai_modele:str, type_modele:str, sampling_rate:int):
    if type_modele not in TYPES_VALIDES:
        logging.warning(f"Le type de modèle {type_modele} n'est pas valide")
        return

    try:
        with get_db_cursor() as cursor:
            cursor.execute("""
                INSERT INTO modele (name, vrai_modele, type_modele, sampling_rate)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    vrai_modele = VALUES(vrai_modele),
                    type_modele = VALUES(type_modele),
                    sampling_rate = VALUES(sampling_rate)
            """, (model, vrai_modele, type_modele, sampling_rate))
    except Exception as e:
        logging.error(f"Erreur lors de l'ajout du modèle {model}: {e}")


def add_all_base_models():
    with open("new_api/services/database/base_models.json", "r", encoding="utf-8") as f:
        base_models = json.load(f)
    for model in base_models:
        ajoute_model(model["name"], model["vrai_modele"], model["type_modele"], model["sampling_rate"])




#DELETE

def delete_model(model:str):
    with get_db_cursor() as cursor:
        cursor.execute("""
            DELETE audio_model_results
            FROM audio_model_results
            INNER JOIN modele ON audio_model_results.id_model = modele.id
            WHERE modele.name = %s
        """, (model,))
        cursor.execute("""
            DELETE results_model
            FROM results_model
            INNER JOIN modele ON results_model.id_model = modele.id
            WHERE modele.name = %s
        """, (model,))
        cursor.execute("DELETE FROM modele WHERE name = %s", (model,))


def reset_models():
    from services.database.audio_results import create_table_results

    with get_db_cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS audio_model_results")
        cursor.execute("DROP TABLE IF EXISTS modele")


    create_table_models()
    create_table_results()