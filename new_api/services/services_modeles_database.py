import mysql.connector


def create_table_models():
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()

    # CREATION DE LA TABLE
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS modele (
                id INT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(100) NOT NULL UNIQUE,
                wer FLOAT
            )
        """)
        conn.commit()
    finally:
        cursor.close()
        conn.close()
    
def ajoute_model(liste_noms_model):


    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()

    # CREATION DE LA TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS modele (
            id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(100) NOT NULL UNIQUE,
            wer FLOAT
        )
    """)
    conn.commit()

    try:
        # Insertion des modèles
        for nom_model in liste_noms_model:
            cursor.execute("""
                INSERT IGNORE INTO modele (name) VALUES (%s)
            """, (nom_model,))
        
        conn.commit()
    finally:
        cursor.close()
        conn.close()



def get_all_models():
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()

    result = []
    try:
        cursor.execute("SELECT name FROM modele")
        result = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()
    
    return result


def delete_all_models():
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM modele")
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def calculate_wer(model):

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
        conn.commit()

    finally:
        cursor.close()
        conn.close()

def calculate_wer_full(app):
    """Calcule les wer de tous les modèles actifs de l'app""" 
    from services.services_results_database import estimer_tous_les_wer, translate_all
    from services.services_models import get_all_active_models 
    
    translate_all(app, True)
    estimer_tous_les_wer()
    
    liste_models = [nom for (nom, _) in get_all_active_models(app)]
    for model in liste_models:
        calculate_wer(model)


