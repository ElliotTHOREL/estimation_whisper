
#CREATE
def load_dataset_interv(deb, fin):
    # Connexion à MariaDB
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()

    # Création de la table avec id manuel
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audio (
        id INT PRIMARY KEY,
        path VARCHAR(255),
        sentence TEXT
    )
    """)
    conn.commit()
    try:
        # Charger le fichier TSV
        path_dataset = os.getenv("PATH_DATASET")
        chemin_tsv = os.path.join(path_dataset, "translated.tsv")

        with open(chemin_tsv, "r", encoding="utf-8") as f:
            next(f)  # sauter l’en-tête
            for i, ligne in enumerate(f):
                if i >= fin:
                    break
                if i >= deb:
                    valeurs = ligne.strip().split("\t")


                    path = valeurs[1]
                    sentence = valeurs[3]

                    cursor.execute("""
                    INSERT IGNORE INTO audio (id, path, sentence)
                    VALUES (%s, %s, %s)
                    """, (i, path, sentence))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def load_dataset(nb_audio):
    load_dataset_interv(0, nb_audio)


#READ
def read_dataset():
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audio")
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result



#DELETE
def delete_dataset():
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM audio")
        conn.commit()

    except mysql.connector.Error as err:
        if err.errno == 1146:  # Code erreur : table n'existe pas
            print("La table 'audio' n'existe pas.")
    finally:
        cursor.close()
        conn.close()

def delete_dataset_interv(deb, fin):
    #Connexion à MariaDB
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM audio WHERE id BETWEEN %s AND %s", (deb, fin))
        conn.commit()

    except mysql.connector.Error as err:
        if err.errno == 1146:  # Code erreur : table n'existe pas
            print("La table 'audio' n'existe pas.")
    finally:
        cursor.close()
        conn.close()
