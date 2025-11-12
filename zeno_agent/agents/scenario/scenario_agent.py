"""
Scenario Analysis Sub-Agent for policy and market impact assessment.


Provides streaming and non-streaming economic scenario analysis
using LLM-powered interpretation of structured data and policy documents.
"""

import os
import re
import traceback
from typing import Dict, Any, Generator
from google import genai
from .scenario_db import build_structured_context
from .scenario_helpers import (
    merge_rag_content,
    encode_query_to_vector,
    build_scenario_prompt,
    query_rag_embeddings_semantic,
)


class ScenarioSubAgent:
    """
    Economist-focused agent for scenario analysis.
    
    Analyzes policy changes, market shocks, and trade interventions
    by combining structured economic data with policy document context.
    """

    def __init__(self, api_key=None):
        """
        Initialize the scenario analysis agent.
        
        Args:
            api_key: Google AI API key (defaults to GOOGLE_API_KEY env var)
            
        Raises:
            EnvironmentError: If API key is not provided or found
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable is required for ScenarioSubAgent."
            )
        self.client = genai.Client(api_key=api_key)

    def extract_entities(self, query: str) -> tuple:
        """
        Extract commodity and country from natural language query.
        
        Args:
            query: User's scenario query text
            
        Returns:
            Tuple of (commodity, country), where either may be None if not found
        """
        commodity_patterns = r"(maize|coffee|tea|oil|wheat|sugar|rice|beans|sorghum)"
        country_patterns = r"(kenya|uganda|tanzania|ethiopia|rwanda|burundi|south sudan)"
        
        commodity_match = re.search(commodity_patterns, query, re.IGNORECASE)
        country_match = re.search(country_patterns, query, re.IGNORECASE)

        commodity = commodity_match.group(1).lower() if commodity_match else None
        country = country_match.group(1).lower() if country_match else None
        
        return commodity, country

    def get_rag_context(self, query: str) -> str:
        """
        Retrieve relevant policy and macroeconomic context using semantic search.
        
        Args:
            query: User's query for semantic matching
            
        Returns:
            Merged text from relevant documents, or fallback message if none found
        """
        try:
            embedding = encode_query_to_vector(query)
            results = query_rag_embeddings_semantic(embedding, top_k=5)
            
            if results:
                return merge_rag_content(results)
        except Exception:
            pass
        
        return "No relevant policy, macroeconomic, or event documents found."

    def run_analysis(self, scenario_query: str, file_context: str = "") -> dict:
        """
        Execute non-streaming scenario analysis.
        
        Args:
            scenario_query: User's scenario description
            file_context: Optional additional context from uploaded files
            
        Returns:
            Dictionary containing analysis results or error information
        """
        return self._execute_analysis(scenario_query, file_context, streaming=False)

    def stream_analysis(
        self,
        scenario_query: str,
        file_context: str = ""
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Execute streaming scenario analysis.
        
        Yields metadata first, then streams LLM-generated analysis chunks,
        and finally signals completion.
        
        Args:
            scenario_query: User's scenario description
            file_context: Optional additional context from uploaded files
            
        Yields:
            Dictionaries containing metadata, content chunks, or completion signals
        """
        yield from self._execute_analysis(scenario_query, file_context, streaming=True)

    def _execute_analysis(
        self,
        scenario_query: str,
        file_context: str = "",
        streaming: bool = False
    ) -> Any:
        """
        Core analysis execution logic for both streaming and non-streaming modes.
        
        Args:
            scenario_query: User's scenario description
            file_context: Optional additional context
            streaming: Whether to stream results or return complete analysis
            
        Returns:
            Generator for streaming mode, dict for non-streaming mode
        """
        query = scenario_query.strip()
        
        # Phase 1: Extract and validate entities
        commodity, country = self.extract_entities(query)

        if not commodity or not country:
            error_response = {
                "type": "error",
                "message": (
                    "Please specify both a commodity (e.g., coffee, maize, tea) "
                    "and a country (e.g., Kenya, Rwanda, Tanzania) in your scenario."
                ),
            }
            if streaming:
                yield error_response
                return
            return error_response

        # Phase 2: Gather context from multiple sources
        try:
            structured_context = build_structured_context(commodity, country)
        except Exception as e:
            structured_context = f"Data retrieval error: {str(e)}"
        
        rag_context = self.get_rag_context(query)
        
        # For streaming: yield metadata before starting LLM generation
        if streaming:
            yield {
                "type": "scenario",
                "status": "SETUP_COMPLETE",
                "entities": {"commodity": commodity, "country": country},
                "data_available": bool(structured_context),
            }

        # Build comprehensive context
        context_parts = []
        if file_context:
            context_parts.append(f"Uploaded document context:\n{file_context[:2000]}")
        if structured_context:
            context_parts.append(f"Economic Data:\n{structured_context}")
        if rag_context:
            context_parts.append(f"Policy Context:\n{rag_context}")
        
        # Phase 3: Generate LLM analysis
        prompt = build_scenario_prompt(query, structured_context, rag_context)
        
        try:
            if streaming:
                # Stream LLM response
                response_stream = self.client.models.generate_content_stream(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                
                for chunk in response_stream:
                    if chunk.text:
                        yield {
                            "type": "scenario_chunk",
                            "content": chunk.text,
                        }
                
                # Signal completion
                yield {
                    "type": "scenario_complete",
                    "status": "STREAM_COMPLETE"
                }
                
            else:
                # Non-streaming response
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                analysis = response.text.strip()
                
                return {
                    "type": "scenario",
                    "commodity": commodity,
                    "country": country,
                    "llm_analysis": analysis,
                    "data_context": structured_context,
                    "followup": "Consider adjusting policy parameters or exploring alternative scenarios."
                }
            
        except Exception as e:
            error_msg = f"Analysis generation failed: {str(e)}"
            error_event = {
                "type": "error",
                "message": error_msg,
                "traceback": traceback.format_exc()
            }
            
            if streaming:
                yield error_event
            else:
                return error_event