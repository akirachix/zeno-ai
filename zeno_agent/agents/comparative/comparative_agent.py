import os
import time
from typing import Any, Dict, List
import google.generativeai as genai
from dotenv import load_dotenv
from zeno_agent.embedding_utils import encode_query_to_vector

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
_GEN_MODEL = None
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    _GEN_MODEL = genai.GenerativeModel("gemini-2.0-flash")

from zeno_agent.db_utils import query_rag_embeddings_semantic

_COMP_CACHE: Dict[str, tuple] = {}
_COMP_CACHE_TTL = 600

def _cache_get(key: str):
    entry = _COMP_CACHE.get(key)
    if entry and (time.time() - entry[0] < _COMP_CACHE_TTL):
        return entry[1]
    _COMP_CACHE.pop(key, None)
    return None

def _cache_set(key: str, val: Any):
    _COMP_CACHE[key] = (time.time(), val)

def merge_rag_content(rag_results: List[Dict]) -> str:
    seen = set()
    merged = []
    for doc in rag_results:
        content = str(doc.get("content", "")).strip()
        if len(content) < 20:
            continue
        key = content[:120]
        if key in seen:
            continue
        seen.add(key)
        merged.append(content)
    return " ".join(merged)

def comparative_search(query: str, top_k: int = 5) -> str:
    try:
        embedding = encode_query_to_vector(query)
        print(f"DEBUG: Raw embedding type: {type(embedding)}")
        
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        if not isinstance(embedding, list):
            embedding = list(embedding)
        embedding = [float(x) for x in embedding]
        print(f"DEBUG: Normalized embedding length: {len(embedding)}")

        raw_results = query_rag_embeddings_semantic(embedding, top_k=top_k)
        return merge_rag_content(raw_results)
    except Exception as e:
        print(f"RAG search failed: {e}")
        return ""

def synthesize_comparative_analysis(query: str, evidence_text: str) -> str:
    cache_key = f"comp::{hash((query, evidence_text))}"
    cached = _cache_get(cache_key)
    if cached:
        return cached
    if not evidence_text:
        fallback = "No relevant trade data found for comparison."
        _cache_set(cache_key, fallback)
        return fallback
    if _GEN_MODEL:
        prompt = f"""You are Zeno, a Senior Economist for East African trade.
User Question: "{query}"
Evidence: {evidence_text}

Provide a short (<=150 words) comparative analysis, start with a one-line conclusion, support with 2-3 facts, and explain WHY.
"""
        try:
            resp = _GEN_MODEL.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(max_output_tokens=250, temperature=0.25)
            )
            text = resp.text.strip()
        except Exception as e:
            text = f"Analysis unavailable (LLM error: {str(e)[:100]}). Evidence: {evidence_text[:200]}..."
    else:
        text = f"[LLM disabled] Evidence: {evidence_text[:300]}..."
    _cache_set(cache_key, text)
    return text

class ComparativeAgent:
    def run(self, payload):
        query = payload.get("query") if isinstance(payload, dict) else str(payload)
        cache_key = f"comparative::{query.lower()}"
        cached = _cache_get(cache_key)
        if cached:
            return {"analysis": cached, "sources": []}
        evidence = comparative_search(query, top_k=5)
        analysis = synthesize_comparative_analysis(query, evidence)
        result = {"analysis": analysis, "evidence": evidence, "sources": []}
        _cache_set(cache_key, result)
        return result

comparative_agent = ComparativeAgent()