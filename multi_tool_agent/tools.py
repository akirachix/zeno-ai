import os
from dotenv import load_dotenv
import psycopg2
from typing import Dict, Any, List, Optional
import traceback
import google.generativeai as genai

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

def embed_text(text: str) -> Optional[List[float]]:
    """
    Get the embedding for the provided text using Gemini's text-embedding-004 model.
    Returns a list of floats (the embedding vector), or None on error.
    """
    try:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        embedding = response["embedding"]
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        traceback.print_exc()
        return None

def query_embeddings(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Perform a semantic similarity search on zeno.rag_embeddings table using pgvector.
    Returns a dictionary with status and results or error message.
    """
    query_vector = embed_text(query)
    if query_vector is None:
        print(" Failed to generate query embedding for input:", query)
        return {
            "status": "error",
            "error_message": "Failed to generate query embedding."
        }

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT embedding_id, content, source, created_at
            FROM zeno.rag_embeddings
            ORDER BY embedding_vector <-> %s::vector
            LIMIT %s;
        """, (query_vector, top_k))

        rows = cur.fetchall()
        results = [
            {
                "embedding_id": row[0],
                "content": row[1],
                "source": row[2],
                "created_at": str(row[3])
            }
            for row in rows
        ]

        cur.close()
        conn.close()
        return {"status": "success", "results": results}
    except Exception as e:
        print(f" Database/query error: {e}")
        traceback.print_exc()
        return {"status": "error", "error_message": str(e)}

