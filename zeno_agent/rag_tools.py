from .db_utils import query_rag_embeddings_semantic
from .embedding_utils import encode_query_to_vector
from typing import List, Dict
from google import genai
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")

client = genai.Client(api_key=GOOGLE_API_KEY)


def summarize_chunk(chunk_text: str, user_query: str) -> str:
    """
    Ask Gemini to summarize a knowledge chunk in the context of the user query.
    This removes irrelevant/trivial text and keeps the answer concise.
    """
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

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.output_text.strip()
    except Exception as e:
        print(f"[Warning] Summarization failed: {e}")
        return chunk_text  


def ask_knowledgebase(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    """
    Retrieve relevant knowledge chunks from zeno.rag_embeddings using semantic search.
    Then summarize/filter each chunk to remove irrelevant/trivial content.
    """
    if not query.strip():
        return [{"content": "Empty query provided.", "source": "N/A"}]

    try:
        embedding = encode_query_to_vector(query)

        results = query_rag_embeddings_semantic(embedding, top_k=top_k)
        if not results:
            return [{"content": "No relevant documents found.", "source": "N/A"}]

        summarized_results = []
        for r in results:
            summary = summarize_chunk(r["content"], query)
            summarized_results.append({
                "content": summary,
                "source": r.get("source", "Unknown")
            })

        return summarized_results

    except Exception as e:
        print(f"[Warning] RAG query failed: {e}")
        return [{"content": f"RAG query failed: {str(e)}", "source": "N/A"}]

def ask_knowledgebase_with_context(query: str, file_context: str = "", top_k: int = 5) -> str:
    base_results = ask_knowledgebase(query, top_k)
    if not file_context:
        return base_results[0]["content"] if base_results else "No info found."
    
    summary = summarize_chunk(file_context, query)
    return f"Uploaded document: {summary}\n\nKnowledge base: {base_results[0]['content']}"
