import os
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai

try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    LOCAL_EMBED_MODEL = None

embedding_cache = TTLCache(maxsize=2000, ttl=3600)
GENAI_MODEL = "text-embedding-004"
_client = None

def _get_client():
    """Initialize and reuse a single Gemini client."""
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
        _client = genai.Client(api_key=api_key)
    return _client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def encode_query_to_vector(query_text: str, mode: str = "query") -> list[float]:
    if not query_text.strip():
        raise ValueError("Empty query text provided.")

    cache_key = f"{mode}:{query_text}"
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]

    client = _get_client()
    try:
        result = client.models.embed_content(
            model=GENAI_MODEL,
            contents=query_text.strip(),
        )
        if hasattr(result, "embeddings") and result.embeddings:
            vector = result.embeddings[0].values
        else:
            raise ValueError(f"Gemini result missing embeddings: {result}")
        embedding_cache[cache_key] = vector
        return vector
    except Exception as e:
        print(f"[Warning] Gemini embedding failed ({e}). Trying local fallback...")
        if LOCAL_EMBED_MODEL:
            vector = LOCAL_EMBED_MODEL.encode(query_text).tolist()
            embedding_cache[cache_key] = vector
            return vector
        raise RuntimeError(f"Failed to generate embedding via Gemini or fallback: {e}")


def encode_vector_for_postgres(vector: list[float]) -> str:
    """
    Converts a Python list of floats to a Postgres-compatible vector string '[x1,x2,...]'.
    """
    if not isinstance(vector, list) or not all(isinstance(x, (float, int)) for x in vector):
        raise ValueError("Input must be a list of floats/ints.")
    return "[" + ",".join(f"{v:.12f}" for v in vector) + "]"
