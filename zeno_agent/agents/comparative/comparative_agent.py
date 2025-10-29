from typing import Dict, Any
from .processing import get_structured_summary, get_rag_evidence, synthesize_comparative_analysis
from .utils import extract_entities

def run(inputs: Dict[str, str], **kwargs) -> Dict[str, Any]:
    query = inputs.get("query", "").strip()
    file_context = inputs.get("file_context", "")
    if not query:
        return {"error": "No query provided."}

    entities = extract_entities(query)
    structured_data = get_structured_summary(query)
    rag_text = get_rag_evidence(query, top_k=5)
    

    full_rag = rag_text
    if file_context:
        full_rag = f"Uploaded document:\n{file_context[:2000]}\n\nPolicy reports:\n{rag_text}"

    llm_analysis = synthesize_comparative_analysis(query, structured_data, full_rag)
    return {
        "type": "comparative",
        "query": query,
        "entities": entities,
        "response": llm_analysis
    }

class ComparativeAgent:
    name = "comparative"

    @staticmethod
    def run(inputs: Dict[str, str], **kwargs) -> Dict[str, Any]:
        return run(inputs, **kwargs)


comparative_agent = ComparativeAgent()
