import mysql.connector.pooling
import os
from contextlib import contextmanager

# Configuration du pool (une seule fois au démarrage)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'admin'),
    'password': os.getenv('DB_PASSWORD', 'pwd'),
    'database': os.getenv('DB_NAME', 'db_audio'),
    'pool_name': 'audio_pool',
    'pool_size': 20,  # Ajustez selon vos besoins
    'pool_reset_session': True,
    'autocommit': False,
    'charset': 'utf8mb4'
}

# Pool global
_pool = None

def initialize_pool():
    global _pool
    if _pool is None:
        _pool = mysql.connector.pooling.MySQLConnectionPool(**DB_CONFIG)
    return _pool

def get_connection():
    if _pool is None:
        initialize_pool()
    return _pool.get_connection()

# Context manager pour gestion automatique des connexions
@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = get_connection()
        yield conn
    finally:
        if conn:
            conn.close()  # Remet dans le pool

@contextmanager
def get_db_cursor(commit=True):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()