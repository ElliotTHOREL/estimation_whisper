from connection import get_db_cursor

def recup_transcri(id_audio, batch_audio, id_model):
    with get_db_cursor() as cursor:
        query = """
            SELECT transcription_result FROM audio_model_results 
            JOIN modele ON audio_model_results.id_model = modele.id
            WHERE audio_model_results.id_audio = %s AND audio_model_results.batch_audio = %s AND modele.name = %s
        """
        cursor.execute(query, (id_audio, batch_audio, id_model))
        result = cursor.fetchone()[0]
        with open("saved_transcriptions.txt", "a", encoding="utf-8") as f:
            f.write(result)
            f.write("\n")
            f.write("\n")



if __name__ == "__main__":
    recup_transcri(0, "Reu", "b-w-large-v3-distil")
    recup_transcri(1, "Reu", "b-w-large-v3-distil")


