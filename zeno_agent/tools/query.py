import os
from dotenv import load_dotenv
import psycopg2
from typing import List, Dict, Any, Optional
import google.generativeai as genai

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

def embed_text(text: str) -> Optional[List[float]]:
    try:
        response = genai.embed_content(model="models/text-embedding-004", content=text.strip())
        return response["embedding"]
    except:
        return None

def query_embeddings(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    query_vector = embed_text(query)
    if not query_vector:
        return []
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT content, source FROM zeno.rag_embeddings
            ORDER BY embedding_vector <-> %s::vector LIMIT %s
        """, (query_vector, top_k))
        results = [{"content": r[0], "source": r[1]} for r in cur.fetchall()]
        cur.close()
        conn.close()
        return results
    except:
        return []