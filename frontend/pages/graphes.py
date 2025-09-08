import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from get import get_models, get_estimations_models, get_types_modeles, get_batch_audios, get_tailles_batch


def plot_graphe_perf(model_names, wer_values, duration_values, model_types):
    # Attribution d'une couleur unique par type de modèle
    unique_types = list(set(model_types))
    color_map = {t: plt.cm.tab10(i) for i, t in enumerate(unique_types)}

    colors = [color_map[t] for t in model_types]

    # Création du nuage de points
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(duration_values, wer_values,
                        s=100, alpha=0.7,
                        c=colors)

    # Ajout de la légende
    for t in unique_types:
        plt.scatter([], [], color=color_map[t], label=t, s=100)
    plt.legend(title="Type de modèle")

    # Ajout des labels pour chaque point
    for i, name in enumerate(model_names):
        plt.annotate(name, 
                    (duration_values[i], wer_values[i]),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.8)

    # Configuration des axes et titre
    plt.xlabel('Durée moyenne (secondes)', fontsize=12)
    plt.ylabel('WER moyen (%)', fontsize=12)
    plt.yscale('log')
    plt.xscale('log')

    ax = plt.gca()
    # Ticks personnalisés
    y_ticks_majors = [10,20,30,50,70,100]
    y_ticks_minors = [8,9,10,20,30,40,60,70,80,90,100]
    x_ticks_majors = [0.1,0.2,0.3,0.5,0.7,1,2,3,5,7,10,20,30,50,70,100]
    x_ticks_minors = [0.2,0.3,0.4,0.6,0.7,0.8,0.9,1,2,3,4,6,7,8,9,10,20,30,40,60,70,80,90,100]

    # Forcer leur position
    ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks_majors))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(y_ticks_minors))
    ax.xaxis.set_major_locator(ticker.FixedLocator(x_ticks_majors))
    ax.xaxis.set_minor_locator(ticker.FixedLocator(x_ticks_minors))

    # Formatter pour afficher les valeurs telles quelles
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


    ax.grid(which='major', color='gray', linestyle='-', alpha=0.3)
    ax.grid(which='minor', color='gray', linestyle='--', alpha=0.2)

    plt.title('Performance des modèles : WER vs Durée de traitement', fontsize=14, fontweight='bold')

    # Grille pour une meilleure lisibilité
    plt.grid(True, alpha=0.3)

    # Ajustement des marges
    plt.tight_layout()
    return plt.gcf()
    
    


def afficher_graphes():
    models = st.session_state["models"]
    estimations_models = st.session_state["estimations_models"]
    selected_ids = [
        [models[id_model]["nom"],100* wer_moyen, duree_moyenne, models[id_model]["type_modele"]]
        for _, id_model, _, _, duree_moyenne, wer_moyen in estimations_models
        if st.session_state.get(f"checkbox_{id_model}", False)
    ]
    model_names, wer_values, duration_values, model_types = zip(*selected_ids)
    fig = plot_graphe_perf(model_names, wer_values, duration_values, model_types)
    st.session_state["fig"] = fig

def form_choix_estim():
    if "models" not in st.session_state:
        get_models()
    models = st.session_state["models"]
    if "estimations_models" not in st.session_state:
        get_estimations_models()
    estimations_models = st.session_state["estimations_models"]





    with st.form("choix_estim"):
        if "default_mode" not in st.session_state:
            st.session_state["default_mode"] = "default"

        if "default_checkbox_state" not in st.session_state:
            st.session_state["default_checkbox_state"] = False

        if "filters_applied" not in st.session_state:
            st.session_state["filters_applied"] = [[], [], []]

        col1,col2,col3 = st.columns([3,1,1])
        with col1:
            if st.form_submit_button("Afficher les graphes"):
                afficher_graphes()
        with col2:
            if st.form_submit_button("Tout cocher"):
                st.session_state["default_mode"] = "default"
                st.session_state["default_checkbox_state"] = True
        with col3:
            if st.form_submit_button("Tout décocher"):
                st.session_state["default_mode"] = "default"
                st.session_state["default_checkbox_state"] = False

        col1, col2, col3, col4 = st.columns([1,1,1,1])
        with col1:
            if "types_modeles" not in st.session_state:
                get_types_modeles()
            types_modeles = st.session_state["types_modeles"]
            selected_types_modeles = st.multiselect("Types modèles", types_modeles)
        with col2:
            if "batch_audios" not in st.session_state:
                get_batch_audios()
            batch_audios = st.session_state["batch_audios"]
            selected_batch_audios = st.multiselect("Batch audio", batch_audios)
        with col3:
            if "tailles_batch" not in st.session_state:
                get_tailles_batch()
            tailles_batch = st.session_state["tailles_batch"]
            selected_tailles_batch = st.multiselect("Taille batch", tailles_batch)
        with col4:
            if st.form_submit_button("Appliquer les filtres"):
                st.session_state["filters_applied"] = [selected_types_modeles, selected_batch_audios, selected_tailles_batch]
                st.session_state["default_mode"] = "filters"



        col1,col2,col3,col4 = st.columns([10,5,2,2])
        with col1:
            st.markdown("##### Modèle")
        with col2:
            st.markdown("##### Batch audio")
        with col3:
            st.markdown("##### Taille batch")
        with col4:
            st.markdown("##### Afficher")

        for _, id_model, nom_batch_audio, size_batch, _, _ in st.session_state["estimations_models"]:
            col1,col2,col3,col4 = st.columns([10,5,2,2])
            with col1:
                st.write(models[id_model]["nom"])
            with col2:
                st.write(nom_batch_audio)
            with col3:
                st.write(size_batch)
            with col4:
                if st.session_state["default_mode"] == "filters":
                    selected_types_modeles, selected_batch_audios, selected_tailles_batch = st.session_state["filters_applied"]
                    satisfied_type = len(selected_types_modeles) ==0 or models[id_model]["type_modele"] in selected_types_modeles
                    satisfied_batch = len(selected_batch_audios) ==0 or nom_batch_audio in selected_batch_audios
                    satisfied_size = len(selected_tailles_batch) ==0 or size_batch in selected_tailles_batch
                    default_value = satisfied_type and satisfied_batch and satisfied_size
                else: #default mode
                    default_value = st.session_state["default_checkbox_state"]
                st.checkbox(label=nom_batch_audio, key=f"checkbox_{id_model}",value=default_value, label_visibility="collapsed")
        

        



def app():
    st.title("Graphes")
    form_choix_estim()
    
    if "fig" in st.session_state:
        st.pyplot(st.session_state["fig"])

if __name__ == "__main__":
    app()