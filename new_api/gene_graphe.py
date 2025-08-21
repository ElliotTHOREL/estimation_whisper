import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


from connection import get_db_cursor


def create_graphe_perf():
    query = """
        SELECT 
            m.name AS model_name,
            rm.wer_moyen AS average_wer,
            rm.duree_moyenne AS average_duration,
            m.type_modele AS model_type
        FROM results_model rm
        JOIN modele m ON rm.id_model = m.id;
        """
    with get_db_cursor() as cursor:
        cursor.execute(query)
        models = cursor.fetchall()
        model_names = [row[0] for row in models]
        wer_values = [100*row[1] for row in models] #en pourcentage
        duration_values = [row[2] for row in models]
        model_types = [row[3] for row in models]

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


        output_folder = "benchmarks"  # Changez selon votre dossier souhaité
        filename = "wer_vs_duration.png"

        os.makedirs(output_folder, exist_ok=True)
        print("plouf")
        filepath = f"{output_folder}/{filename}"


        plt.savefig(filepath, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    create_graphe_perf()






