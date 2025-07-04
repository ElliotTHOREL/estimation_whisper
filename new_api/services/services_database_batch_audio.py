import mysql.connector



#CREATE
def create_table_batch_audio():
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS batch_audio (
        id INT PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(100) NOT NULL UNIQUE,
        path VARCHAR(255) NOT NULL UNIQUE
    )
    """)
    conn.commit()


#READ
def get_all_batch_audio():
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()

    result = None
    try:
        cursor.execute("SELECT name, path FROM batch_audio")
        result = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()
    return result


#UPDATE
def add_batch_audio(id, name, path):
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO batch_audio (name, path) VALUES (%s, %s)", (name, path))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


#DELETE
def delete_batch_audio(name):
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM batch_audio WHERE name = %s", (name,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def delete_table():
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="admin",
        password="pwd",
        database="db_audio"
    )
    cursor = conn.cursor()
    try:
        cursor.execute("DROP TABLE batch_audio")
        conn.commit()
    finally:
        cursor.close()
        conn.close()
    