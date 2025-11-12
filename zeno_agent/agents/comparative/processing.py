"""
Processing utilities for comparative trade analysis.

Handles data retrieval, summarization, RAG context enrichment,
and LLM-based analysis generation with streaming support.

ENHANCED VERSION - Ready to replace your current processing.py
"""

import time
import random
import pandas as pd
from typing import Dict, Any, Generator
from .utils import (
    client,
    extract_entities,
    calculate_cagr,
    merge_rag_content,
    encode_query_to_vector,
)
from zeno_agent.db_utils import (
    get_country_id_by_name,
    get_product_id_by_name,
    get_indicator_id_by_metric,
    get_trade_data_from_db,
    query_rag_embeddings_semantic,
)
from google.genai import errors


def summarize_trade_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate high-level summary metrics from trade data.
    
    Returns aggregated statistics without exposing raw data points.
    
    Args:
        df: Trade data DataFrame with 'date', 'quantity', and 'price' columns
        
    Returns:
        Dictionary containing summary text and CAGR metric
    """
    if df.empty:
        return {"summary": "No trade data available.", "cagr": 0.0}

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = df.dropna(subset=["quantity", "price"])
    
    if df.empty:
        return {"summary": "No valid trade data after cleaning.", "cagr": 0.0}

    yearly = df.groupby("year").agg({"quantity": "sum", "price": "mean"}).reset_index()
    start_year, end_year = yearly["year"].min(), yearly["year"].max()
    total_quantity = yearly["quantity"].sum()
    avg_price = yearly["price"].mean()

    if len(yearly) > 1:
        cagr = calculate_cagr(
            float(yearly.iloc[0]["quantity"]),
            float(yearly.iloc[-1]["quantity"]),
            end_year - start_year,
        )
    else:
        cagr = 0.0

    return {
        "summary": (
            f"Period {start_year}-{end_year}: "
            f"Total volume {total_quantity:,.0f} units, "
            f"Average price KES {avg_price:,.2f}, "
            f"CAGR {cagr:.2f}%"
        ),
        "cagr": cagr,
    }


def get_structured_summary(query: str) -> Dict[str, str]:
    """
    Extract entities from query and retrieve summarized trade data for each country.
    
    Args:
        query: User's natural language query
        
    Returns:
        Dictionary mapping country names to their trade data summaries
    """
    entities = extract_entities(query)
    countries = entities.get("countries", [])
    commodity = entities.get("commodity", "")
    
    if not countries or not commodity:
        return {}
    
    results = {}
    for country in countries:
        try:
            country_id = get_country_id_by_name(country)
            product_id = get_product_id_by_name(commodity)
            indicator_id = get_indicator_id_by_metric("exports")
            
            df = get_trade_data_from_db(country_id, product_id, indicator_id)
            summary = summarize_trade_data(df)
            results[country] = summary["summary"]
            
        except Exception as e:
            results[country] = f"Data unavailable: {str(e)}"
            
    return results


def get_rag_evidence(query: str, top_k: int = 5) -> str:
    """
    Retrieve relevant policy documents and reports using semantic search.
    
    Args:
        query: User's query for semantic matching
        top_k: Number of top relevant documents to retrieve
        
    Returns:
        Merged text content from retrieved documents
    """
    try:
        embedding = encode_query_to_vector(query)
        raw_results = query_rag_embeddings_semantic(embedding, top_k=top_k)
        return merge_rag_content(raw_results) if raw_results else ""
    except Exception:
        return ""


def _generate_analysis_prompt(
    query: str,
    structured_data: Dict[str, str],
    rag_text: str
) -> str:
    """
    Construct comprehensive analysis prompt for LLM.
    
    Creates a structured prompt that guides the LLM to produce
    professional comparative economic analysis.
    
    Args:
        query: Original user query
        structured_data: Country-wise trade data summaries
        rag_text: Contextual evidence from policy documents
        
    Returns:
        Formatted prompt string for LLM
    """
    countries = list(structured_data.keys())
    num_countries = len(countries)
    
    context_lines = [
        f"**{country}**: {summary}"
        for country, summary in structured_data.items()
    ]
    data_context = "\n".join(context_lines)
    
    rag_section = ""
    if rag_text:
        rag_section = f"""
=== CONTEXTUAL EVIDENCE FROM POLICY REPORTS ===
{rag_text}
"""

    return f"""You are a senior trade economist at an East African regional economic commission specializing in comparative trade analysis. Produce a comprehensive comparative report based on rigorous economic analysis.

=== ANALYSIS SPECIFICATION ===
User Query: "{query}"
Countries Under Analysis: {', '.join(countries)} ({num_countries} countries)
Analysis Type: Comparative trade performance assessment

=== TRADE DATA SUMMARIES ===
{data_context}
{rag_section}

=== ANALYTICAL FRAMEWORK ===
Your analysis must follow this structure using clear, professional paragraphs (no bullet points, lists, or markdown formatting):

**Paragraph 1: Executive Overview**
Open with a clear statement of the comparative question being addressed. Provide a high-level summary of the key finding: which country demonstrates the strongest trade performance and why. Reference specific metrics from the data summaries to support this conclusion. Set the context for the detailed analysis that follows.

**Paragraph 2: Quantitative Comparative Analysis**
Systematically compare trade performance across all {num_countries} countries using the provided metrics. Analyze total trade volumes, average prices, and compound annual growth rates (CAGR). Identify leaders and laggards in each dimension. Explain what the differences in CAGR reveal about growth trajectories and competitive dynamics. Note any data limitations or gaps that affect the comparison.

**Paragraph 3: Structural and Competitive Drivers**
Examine the underlying factors that explain the observed performance differences. Consider production capacity, quality standards, market access advantages, value chain positioning, and policy environments. Reference any relevant insights from the policy reports context. Discuss how regional trade agreements (COMESA, EAC, AfCFTA) may differentially impact the countries. Identify structural strengths and weaknesses in each country's trade profile.

**Paragraph 4: Market Context and External Factors**
Assess how global market conditions, regional demand patterns, and external shocks may be affecting comparative performance. Consider currency fluctuations, competing origins, demand trends in key destination markets, and supply chain disruptions. Explain how these factors may amplify or mitigate inherent competitive advantages.

**Paragraph 5: Strategic Implications and Recommendations**
Provide actionable insights for different stakeholders in each country. For the leading performer, suggest strategies to maintain advantage. For countries lagging behind, recommend specific interventions to improve competitiveness. Address policy implications for regional trade facilitation and harmonization. Consider opportunities for countries to specialize in different market segments or quality tiers to reduce direct competition.

=== WRITING STANDARDS ===
- Write in clear, flowing paragraphs with smooth transitions between sections
- Use precise numerical references from the data summaries
- Maintain analytical objectivity and avoid promotional language
- Support all claims with evidence from the data or contextual reports
- Use appropriate economic terminology (terms of trade, revealed comparative advantage, trade intensity, etc.)
- Target audience: Trade policymakers, commodity board executives, regional economic analysts
- Length: Comprehensive but focused; approximately 600-800 words
- Tone: Professional, analytical, forward-looking

Begin your analysis now:"""


def synthesize_comparative_analysis(
    query: str,
    structured_data: Dict[str, str],
    rag_text: str
) -> str:
    """
    Generate comparative analysis using non-streaming LLM call.
    
    Args:
        query: User's original query
        structured_data: Country-wise trade summaries
        rag_text: RAG-retrieved contextual evidence
        
    Returns:
        Complete analysis text
    """
    prompt = _generate_analysis_prompt(query, structured_data, rag_text)
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Analysis generation failed: {str(e)}"


def stream_comparative_analysis(
    query: str,
    structured_data: Dict[str, str],
    rag_text: str
) -> Generator[str, None, None]:
    """
    Generate comparative analysis using streaming LLM call with retry logic.
    
    Implements exponential backoff to handle API rate limits gracefully.
    
    Args:
        query: User's original query
        structured_data: Country-wise trade summaries
        rag_text: RAG-retrieved contextual evidence
        
    Yields:
        Text chunks as they become available from the LLM
        
    Raises:
        RuntimeError: If max retries exceeded due to rate limits
    """
    prompt = _generate_analysis_prompt(query, structured_data, rag_text)
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response_stream = client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            
            for chunk in response_stream:
                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text
            return
            
        except errors.ClientError as e:
            error_msg = str(e)
            if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(
                        "Failed to generate analysis after multiple retries due to API rate limits. "
                        "Please try again in a few moments."
                    )
            raise
            
        except Exception as e:
            raise RuntimeError(f"Unexpected error during analysis generation: {str(e)}")
    
    raise RuntimeError("Failed to generate analysis after maximum retry attempts.")