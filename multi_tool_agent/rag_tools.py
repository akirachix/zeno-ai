from .db_utils import query_rag_embeddings_semantic
from .embedding_utils import encode_query_to_vector
from typing import List, Dict

def ask_knowledgebase(query: str) -> List[Dict[str, str]]:
    """
    Retrieve relevant knowledge chunks from zeno.rag_embeddings using semantic search.
    """
    if not query.strip():
        return [{"content": "Empty query provided.", "source": "N/A"}]
    
    try:
        embedding = encode_query_to_vector(query)
        results = query_rag_embeddings_semantic(embedding)
        if not results:
            return [{"content": "No relevant documents found.", "source": "N/A"}]
        return results
    except Exception as e:
        print(f"RAG query failed: {e}")
        return [{"content": f"RAG query failed: {str(e)}", "source": "N/A"}]