import os
from dotenv import load_dotenv
import psycopg2
from typing import List, Dict, Any, Optional
from google import genai  

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None


def embed_text(text: str) -> Optional[List[float]]:
    """Generate embedding vector for a given text using the new genai client."""
    if not client:
        print(" GOOGLE_API_KEY not set or client not initialized.")
        return None
    try:
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=text.strip()
        )
        return response.embedding
    except Exception as e:
        print(f"[Error] Embedding failed: {e}")
        return None


def query_embeddings(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform semantic similarity search on PostgreSQL pgvector table."""
    query_vector = embed_text(query)
    if not query_vector:
        return []
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT content, source 
            FROM zeno.rag_embeddings
            ORDER BY embedding_vector <-> %s::vector 
            LIMIT %s
            """,
            (query_vector, top_k)
        )
        results = [{"content": r[0], "source": r[1]} for r in cur.fetchall()]
        cur.close()
        conn.close()
        return results
    except Exception as e:
        print(f"[Error] Database query failed: {e}")
        return []
