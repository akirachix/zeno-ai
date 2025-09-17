import google.generativeai as genai
import os
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential

embedding_cache = TTLCache(maxsize=1000, ttl=3600)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def encode_query_to_vector(query_text: str) -> list:
    """
    Uses Gemini's embedding API (text-embedding-004).
    Returns a list of floats representing the embedding vector.
    """
    if not query_text.strip():
        raise ValueError("Empty query text provided.")
    
    cache_key = query_text
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
    genai.configure(api_key=api_key)
    
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query_text,
            task_type="retrieval_query"
        )
        embedding_cache[cache_key] = result["embedding"]
        return result["embedding"]
    except Exception as e:
        raise ValueError(f"Failed to generate embedding: {str(e)}")