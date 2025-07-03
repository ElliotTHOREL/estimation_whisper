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
                name VARCHAR(100) NOT NULL UNIQUE
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
            name VARCHAR(100) NOT NULL UNIQUE
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