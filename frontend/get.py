import requests
from config import API_BASE
import streamlit as st


def get_models():
    url = f"{API_BASE}/modeles_database/all_details"
    response = requests.get(url)
    models = response.json()
    dico_models={}

    for id,nom_court,_,type_modele,_ in models:
        dico_models[id] = {"nom": nom_court, "type_modele": type_modele}
    
    st.session_state["models"] = dico_models


def get_estimations_models():
    url = f"{API_BASE}/database_models_results/"
    response = requests.get(url)
    estimations_models = response.json()
    st.session_state["estimations_models"] = estimations_models

def get_types_modeles():
    url = f"{API_BASE}/modeles_database/types_valides"
    response = requests.get(url)
    types_modeles = response.json()
    st.session_state["types_modeles"] = types_modeles

def get_batch_audios():
    url = f"{API_BASE}/batch_audio_database/"
    response = requests.get(url)
    batch_audios = response.json()
    st.session_state["batch_audios"] = batch_audios

def get_tailles_batch():
    if "estimations_models" not in st.session_state:
        get_estimations_models()
    estimations_models = st.session_state["estimations_models"]
    tailles_batch = list(set([size_batch for _, _, _, size_batch, _, _ in estimations_models]))
    st.session_state["tailles_batch"] = tailles_batch


