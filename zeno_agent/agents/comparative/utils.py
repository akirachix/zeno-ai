import os
from google import genai
from typing import List, Dict, Any


from zeno_agent.embedding_utils import encode_query_to_vector

# CONFIG
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")

client = genai.Client(api_key=GOOGLE_API_KEY)


def extract_entities(query: str) -> Dict[str, Any]:
    country_names = ["kenya", "tanzania", "uganda", "rwanda", "ethiopia", "burundi"]
    commodity_names = ["coffee", "maize", "tea", "beans", "wheat", "rice", "sugar"]

    query_lower = query.lower()
    countries = [c.title() for c in country_names if c in query_lower]
    commodities = [c for c in commodity_names if c in query_lower]

    return {
        "countries": countries[:2],
        "commodity": commodities[0] if commodities else "coffee"
    }


def calculate_cagr(start_value: float, end_value: float, periods: int) -> float:
    if start_value <= 0 or end_value <= 0 or periods <= 0:
        return 0.0
    return ((end_value / start_value) ** (1 / periods) - 1) * 100


def merge_rag_content(rag_results: List[Dict[str, str]]) -> str:
    seen = set()
    merged_content = []
    for doc in rag_results:
        content = doc.get("content", "").strip()
        if len(content) < 20:
            continue
        key = content[:100]
        if key in seen:
            continue
        seen.add(key)
        merged_content.append(content)
    return " ".join(merged_content)
