from .db_utils import query_rag_embeddings_semantic
from .embedding_utils import encode_query_to_vector
from typing import List, Dict
import google.generativeai as genai
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def _get_summarizer_model():
    """Lazy-initialize the summarizer model."""
    if not GOOGLE_API_KEY:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel("gemini-2.0-flash")

def summarize_chunk(chunk_text: str, user_query: str) -> str:
    try:
        model = _get_summarizer_model()
        prompt = f"""
You are a helpful AI assistant. Summarize the following text in 1-2 sentences
to answer the user query. Only include relevant information; discard off-topic content.

User Query: "{user_query}"

Text to summarize:
\"\"\"
{chunk_text}
\"\"\"

Return only the summary.
""".strip()
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(max_output_tokens=150, temperature=0.2)
        )
        return response.text.strip()
    except Exception as e:
        print(f"[Warning] Summarization failed: {e}")
        return chunk_text  

def ask_knowledgebase(query: str, top_k: int = 5) -> str:
    if not isinstance(query, str) or not query.strip():
        return "Empty query provided."
    try:
        embedding = encode_query_to_vector(query)
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        elif not isinstance(embedding, list):
            embedding = list(embedding)
        embedding = [float(x) for x in embedding]

        results = query_rag_embeddings_semantic(embedding, top_k=top_k)
        if not results:
            return "No relevant documents found."

        summaries = []
        for r in results:
            summary = summarize_chunk(r["content"], query)
            source = r.get("source", "Unknown")
            summaries.append(f"{summary} (Source: {source})")
        return "\n\n".join(summaries)
    except Exception as e:
        print(f"[Warning] RAG query failed: {e}")
        return f"RAG query failed: {str(e)}"