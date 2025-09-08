import streamlit as st
import requests
from get import API_BASE, get_models, get_estimations_models, get_batch_audios

if "models" not in st.session_state:
    get_models()
models = st.session_state["models"]
if "estimations_models" not in st.session_state:
    get_estimations_models()
estimations_models = st.session_state["estimations_models"]
if "batch_audios" not in st.session_state:
    get_batch_audios()
batch_audios = st.session_state["batch_audios"]


def appel_ajout_estimation(models, batch_audio, nombre_estim):
    if len(models) == 0:
        url = f"{API_BASE}/database_models_results/all"
        params = {"nom_batch": batch_audio, "taille_echantillon": nombre_estim}
        response = requests.post(url, params=params)
    else:
        for model in models:
            url = f"{API_BASE}/database_models_results/"
            params = {"model": model, "nom_batch": batch_audio, "taille_echantillon": nombre_estim}
            response = requests.post(url, params=params)
    get_estimations_models()
    st.rerun()



def supprimer_estimation(id):
    url = f"{API_BASE}/database_models_results/"
    param = {"id": id}
    response = requests.delete(url, params=param)
    get_estimations_models()
    st.rerun()



def afficher_estimations():
    col1,col2,col3,col4 = st.columns([10,5,2,2])
    with col1:
        st.markdown("##### Modèle")
    with col2:
        st.markdown("##### Batch audio")
    with col3:
        st.markdown("##### Taille batch")
    with col4:
        st.markdown("##### Afficher")


    for id_estimation, id_model, nom_batch_audio, size_batch, _, _ in st.session_state["estimations_models"]:
        col1,col2,col3,col4 = st.columns([10,5,2,2])
        with col1:
            st.write(models[id_model]["nom"])
        with col2:
            st.write(nom_batch_audio)
        with col3:
            st.write(size_batch)
        with col4:
            if st.button("Supprimer", key=f"supprimer_{id_estimation}"):
                supprimer_estimation(id_estimation)

def ajout_estimation():
    models = st.session_state["models"]
    batch_audios = st.session_state["batch_audios"]
    st.markdown("### Effectuer de nouvelles estimations")
    with st.form("ajout_estimation_form"):
        col1, col2, col3, col4 = st.columns([2,2,2,1])
        with col1:
            selected_models=st.multiselect("Modèles", [model["nom"]for model in models.values()])
        with col2:
            selected_batch_audio=st.selectbox("Batch audio", batch_audios)
        with col3:
            nombre_estim=st.number_input("Nombre d'estimations", min_value=1, max_value=1000, value=10)
        with col4:
            if st.form_submit_button():
                appel_ajout_estimation(selected_models, selected_batch_audio, nombre_estim)




def app():
    st.title("Gestion des estimations")
    afficher_estimations()
    ajout_estimation()


if __name__ == "__main__":
    app()