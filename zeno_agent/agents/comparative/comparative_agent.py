"""
Comparative Analysis Agent for multi-country trade comparisons.

Provides both streaming and non-streaming interfaces for generating
comparative economic reports across multiple countries and commodities.
"""

import traceback
from typing import Dict, Any, Generator
from .processing import (
    get_structured_summary,
    get_rag_evidence,
    synthesize_comparative_analysis,
    stream_comparative_analysis
)
from .utils import extract_entities


def run(inputs: Dict[str, str], **kwargs) -> Dict[str, Any]:
    """
    Non-streaming comparative analysis execution.
    
    Args:
        inputs: Dictionary containing 'query' and optional 'file_context'
        
    Returns:
        Dictionary with analysis results or error information
    """
    query = inputs.get("query", "").strip()
    file_context = inputs.get("file_context", "")
    
    if not query:
        return {"error": "No query provided."}

    try:
        entities = extract_entities(query)
        
        if not entities.get("countries") or not entities.get("commodity"):
            return {
                "error": "Could not identify required entities (countries and commodity) from query."
            }
        
        structured_data = get_structured_summary(query)
        rag_text = get_rag_evidence(query, top_k=5)
        
        full_rag = rag_text
        if file_context:
            full_rag = f"Uploaded document context:\n{file_context[:2000]}\n\nPolicy reports:\n{rag_text}"
        
        analysis = synthesize_comparative_analysis(query, structured_data, full_rag)
        
        return {
            "type": "comparative",
            "query": query,
            "entities": entities,
            "analysis": analysis,
            "data_sources": len(structured_data),
            "countries_analyzed": list(structured_data.keys())
        }
        
    except Exception as e:
        return {
            "type": "error",
            "error": f"Comparative analysis failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


def stream(inputs: Dict[str, str], **kwargs) -> Generator[Dict[str, Any], None, None]:
    """
    Streaming comparative analysis execution.
    
    Yields metadata first, then streams the LLM-generated analysis in chunks,
    and finally signals completion.
    
    Args:
        inputs: Dictionary containing 'query' and optional 'file_context'
        
    Yields:
        Dictionaries containing either metadata, content chunks, or completion signals
    """
    query = inputs.get("query", "").strip()
    file_context = inputs.get("file_context", "")
    
    if not query:
        yield {"type": "error", "message": "No query provided."}
        return

    try:
        # Phase 1: Extract entities and validate
        entities = extract_entities(query)
        
        if not entities.get("countries") or not entities.get("commodity"):
            yield {
                "type": "error",
                "message": "Could not identify required entities (countries and commodity) from query."
            }
            return
        
        # Phase 2: Gather structured data and context
        structured_data = get_structured_summary(query)
        
        if not structured_data:
            yield {
                "type": "error",
                "message": "No trade data available for the specified countries and commodity."
            }
            return
        
        rag_text = get_rag_evidence(query, top_k=5)
        
        full_rag = rag_text
        if file_context:
            full_rag = f"Uploaded document context:\n{file_context[:2000]}\n\nPolicy reports:\n{rag_text}"
        
        # Phase 3: Yield initial metadata
        yield {
            "type": "comparative",
            "query": query,
            "entities": entities,
            "countries_analyzed": list(structured_data.keys()),
            "data_sources": len(structured_data),
            "status": "SETUP_COMPLETE"
        }
        
        # Phase 4: Stream the LLM-generated analysis
        analysis_stream = stream_comparative_analysis(query, structured_data, full_rag)
        
        for chunk in analysis_stream:
            yield {
                "type": "comparative_chunk",
                "content": chunk
            }
        
        # Phase 5: Signal completion
        yield {
            "type": "comparative_complete",
            "status": "STREAM_COMPLETE"
        }
        
    except Exception as e:
        yield {
            "type": "error",
            "message": f"Comparative analysis stream failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


class ComparativeAgent:
    """
    Agent for generating comparative trade analysis across multiple countries.
    
    Supports both streaming and non-streaming execution modes.
    """
    
    name = "comparative"
    description = "Analyzes and compares trade performance across multiple countries"
    
    @staticmethod
    def run(inputs: Dict[str, str], **kwargs) -> Dict[str, Any]:
        """Standard non-streaming execution method."""
        return run(inputs, **kwargs)
    
    @staticmethod
    def stream(inputs: Dict[str, str], **kwargs) -> Generator[Dict[str, Any], None, None]:
        """Streaming execution method for real-time response generation."""
        return stream(inputs, **kwargs)


# Export the agent instance
comparative_agent = ComparativeAgent()