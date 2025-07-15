import matplotlib.pyplot as plt


from connection import get_db_cursor


def create_graphe_perf():
    query = """
        SELECT 
            m.name AS model_name,
            rm.wer_moyen AS average_wer,
            rm.duree_moyenne AS average_duration
        FROM results_model rm
        JOIN modele m ON rm.id_model = m.id;
        """
    with get_db_cursor() as cursor:
        cursor.execute(query)
        models = cursor.fetchall()
        model_names = [row[0] for row in models]
        wer_values = [row[1] for row in models]
        duration_values = [row[2] for row in models]

        # Création du nuage de points
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(duration_values, wer_values, 
                            s=100, alpha=0.7, 
                            c=range(len(model_names)), 
                            cmap='viridis')

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
        plt.title('Performance des modèles : WER vs Durée de traitement', fontsize=14, fontweight='bold')

        # Grille pour une meilleure lisibilité
        plt.grid(True, alpha=0.3)

        # Ajustement des marges
        plt.tight_layout()


        output_folder = "benchmarks"  # Changez selon votre dossier souhaité
        filename = "wer_vs_duration.png"
        filepath = f"{output_folder}/{filename}"


        plt.savefig(filepath, dpi=300, bbox_inches='tight')









