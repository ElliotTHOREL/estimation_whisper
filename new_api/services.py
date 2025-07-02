import torch
import logging
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def transcribe(audio_path: str, language: str = None):
    pass


def load_model(model, language):
    pass


def load_dataset_interv(deb, fin):
    load_dotenv()

    path_dataset = os.getenv("PATH_DATASET")
    with open(path_dataset + "/translated.tsv", "r", encoding="utf-8") as f:
        next(f)
        for i, ligne in enumerate(f):
            if i >= fin:
                break
            if i >= deb:
                ligne


                


def load_dataset(nb_audio):
    load_dataset_interv(0, nb_audio)



"""
class Audio: 
    def __init__(self, string_de_metadonnees: str):
        pass
"""
