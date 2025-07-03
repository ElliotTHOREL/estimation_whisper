import mysql.connector


def ajoute_result(model, id_audio, transcription_result, replace):
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()



    try:
        # CREATION DE LA TABLE
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audio_model_results (
                id_audio INT NOT NULL,
                id_model INT NOT NULL,
                transcription_result TEXT,
                PRIMARY KEY (id_audio, id_model),
                FOREIGN KEY (id_audio) REFERENCES audio(id) ON DELETE CASCADE,
                FOREIGN KEY (id_model) REFERENCES modele(id) ON DELETE CASCADE
            )
        """)
        conn.commit()

        # INSERTION DES RESULTATS
        if replace:
            cursor.execute("""
                INSERT INTO audio_model_results (id_audio, id_model, transcription_result)
                VALUES (
                    %s,
                    (SELECT id FROM modele WHERE name = %s),
                    %s
                )
                ON DUPLICATE KEY UPDATE
                    transcription_result = VALUES(transcription_result)
            """, (id_audio, model, transcription_result))
            conn.commit()
        else:
            cursor.execute("""
                INSERT IGNORE INTO audio_model_results (id_audio, id_model, transcription_result)
                VALUES (
                    %s,
                    (SELECT id_model FROM modele WHERE name = %s),
                    %s
                )
            """, (id_audio, model, transcription_result))
            conn.commit()

    
    finally:
        cursor.close()
        conn.close()
