import os
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
genai.configure(api_key=GOOGLE_API_KEY)

embedding_cache = TTLCache(maxsize=2000, ttl=3600)
GENAI_MODEL = "text-embedding-004"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def encode_query_to_vector(query_text: str, mode: str = "query") -> list[float]:
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("Query must be a non-empty string")
    clean_text = query_text.strip()
    cache_key = f"{mode}:{clean_text}"
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]

    try:
        result = genai.embed_content(GENAI_MODEL, clean_text)
    except Exception as e:
        try:
            result = genai.embed_content("embedding-001", clean_text)
        except Exception:
            raise RuntimeError(f"Google embedding failed: {e}")

    vector = None
    if hasattr(result, 'embedding'):
        vector = result.embedding.values
    elif isinstance(result, dict):
        if "embedding" in result:
            embedding_data = result["embedding"]
            if isinstance(embedding_data, dict) and "values" in embedding_data:
                vector = embedding_data["values"]
            elif isinstance(embedding_data, list):
                vector = embedding_data
            else:
                raise ValueError(f"Unexpected embedding data format: {type(embedding_data)}")
        else:
            raise ValueError("Dict result missing 'embedding' key")
    else:
        raise ValueError(f"Unexpected result type: {type(result)}")

    if not isinstance(vector, list):
        raise ValueError(f"Vector is not a list: {type(vector)}")
    if not vector:
        raise ValueError("Empty embedding vector")
    
    try:
        vector = [float(x) for x in vector]
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to convert vector to floats: {e}")

    embedding_cache[cache_key] = vector
    return vector

def encode_vector_for_postgres(vector: list[float]) -> str:
    if not isinstance(vector, list) or not all(isinstance(x, (float, int)) for x in vector):
        raise ValueError("Input must be a list of floats/ints.")
    return "[" + ",".join(f"{v:.12f}" for v in vector) + "]"