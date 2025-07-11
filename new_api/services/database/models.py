from connection import get_db_cursor




def create_table_models():
    with get_db_cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS modele (
                id INT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(100) NOT NULL UNIQUE,
                duree_chargement FLOAT
            )
        """)

    
def ajoute_model(model, temps_chargement):
    with get_db_cursor() as cursor:
        # Vérifier si le modèle existe déjà
        cursor.execute("SELECT COUNT(*) FROM modele WHERE name = %s", (model,))
        exists = cursor.fetchone()[0] > 0

        if not exists:
            # Insérer seulement si n'existe pas
            cursor.execute("""
                INSERT INTO modele (name, duree_chargement) VALUES (%s, %s)
            """, (model, temps_chargement))
        else:
            # Mettre à jour le temps de chargement si existe
            cursor.execute("""
                UPDATE modele SET duree_chargement = %s WHERE name = %s
            """, (temps_chargement, model))
        



def get_all_models():
    with get_db_cursor() as cursor:
        cursor.execute("SELECT name FROM modele")
        return cursor.fetchall()
    



def delete_all_models():
    with get_db_cursor() as cursor:
        cursor.execute("DELETE FROM modele")


def calculate_wer(model):

    with get_db_cursor() as cursor:
        cursor.execute("""
            SELECT AVG(audio_model_results.wer) FROM audio_model_results
            JOIN modele ON audio_model_results.id_model = modele.id
            WHERE modele.name = %s AND audio_model_results.wer IS NOT NULL
        """, (model,))
        avg_wer = cursor.fetchone()[0]

        # Insérer ou mettre à jour la moyenne dans une table dédiée (par exemple, modele_wer)
        # On suppose qu'il existe une table 'modele_wer' avec les colonnes id_model et avg_wer
        cursor.execute("""
            UPDATE modele
            SET wer = %s
            WHERE name = %s
        """, (avg_wer, model))



def calculate_wer_full(app):
    """Calcule les wer de tous les modèles actifs de l'app""" 
    from services.database.results import estimer_tous_les_wer, translate_all
    from services.models import get_all_active_models 
    
    translate_all(app, False)
    estimer_tous_les_wer()
    
    liste_models = [nom for (nom, _) in get_all_active_models(app)]
    for model in liste_models:
        calculate_wer(model)


def reset_models():
    from services.database.results import create_table_results

    with get_db_cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS audio_model_results")
        cursor.execute("DROP TABLE IF EXISTS modele")


    create_table_models()
    create_table_results()