"""
Helper utilities for scenario analysis.

Provides RAG content merging, query encoding, and LLM prompt construction
for scenario-based economic analysis.
"""

from typing import List, Dict, Any
from zeno_agent.embedding_utils import encode_query_to_vector
from zeno_agent.db_utils import query_rag_embeddings_semantic


def merge_rag_content(rag_results: List[Dict[str, Any]]) -> str:
    """
    Merge and deduplicate content from RAG search results.
    
    Args:
        rag_results: List of document dictionaries with 'content' field
        
    Returns:
        Space-joined string of deduplicated content
    """
    seen = set()
    merged = []
    
    for doc in rag_results:
        content = doc.get("content", "").strip()
        
        # Skip very short snippets
        if len(content) < 20:
            continue
        
        # Use first 100 chars as deduplication key
        key = content[:100]
        if key in seen:
            continue
        
        seen.add(key)
        merged.append(content)
    
    return " ".join(merged)


def build_scenario_prompt(
    query: str,
    structured_context: str,
    rag_context: str
) -> str:
    """
    Construct comprehensive LLM prompt for scenario analysis.
    
    Creates a structured prompt that guides the LLM to produce
    professional economic scenario analysis with clear sections.
    
    Args:
        query: User's scenario query
        structured_context: Economic data context from database
        rag_context: Policy and document context from RAG
        
    Returns:
        Formatted prompt string for LLM
    """
    return f"""You are Dr. Zeno, Senior Economist at the East African Trade and Development Institute, specializing in policy impact assessment and market scenario analysis.

Provide a comprehensive professional economic analysis of the following policy or market scenario.

=== SCENARIO SPECIFICATION ===
Query: "{query}"

=== AVAILABLE ECONOMIC DATA ===
Structured Trade & Macro Indicators:
{structured_context}

Policy & Contextual Documents:
{rag_context}

=== ANALYTICAL FRAMEWORK ===
Structure your analysis into five distinct sections, written in clear, flowing paragraphs (no bullet points, lists, or markdown formatting):

**Section 1: Scenario Overview and Context (2-3 sentences)**
Begin by restating the scenario clearly. Establish the baseline economic context using the provided data (current prices, volumes, macro indicators). Identify the key economic agents and markets that will be directly affected.

**Section 2: Immediate Direct Effects (3-4 sentences)**
Analyze the first-order impacts of this scenario. If it involves a policy change (tariff, subsidy, export ban), explain how it directly affects prices, costs, or market access. If it's a shock (drought, price spike, currency movement), describe the immediate transmission mechanism. Use economic terminology: demand elasticity, supply response, price transmission, arbitrage conditions. Reference specific data points from the structured context to ground your analysis.

**Section 3: Price Dynamics and Market Adjustments (3-4 sentences)**
Examine how prices will adjust across the value chain. Consider producer prices, wholesale prices, consumer prices, and export prices. Discuss whether the scenario creates inflationary or deflationary pressure. Explain substitution effects (will consumers or producers switch to alternatives?), inventory responses, and potential for hoarding or dumping. Reference historical price levels from the data to estimate magnitude of changes.

**Section 4: Trade Effects and Regional Implications (3-4 sentences)**
Assess impacts on trade flows: exports, imports, and intra-regional trade within EAC/COMESA. Consider competitiveness effects, trade diversion, and arbitrage opportunities across borders. Discuss whether the scenario strengthens or weakens the country's trade position. If relevant, note implications for foreign exchange earnings, trade balance, and currency pressure.

**Section 5: Macroeconomic Consequences and Policy Recommendations (3-4 sentences)**
Evaluate broader economic impacts: GDP effects, employment (especially in agriculture and rural areas), government revenue (if taxes are involved), and household welfare (food security, purchasing power). Identify who wins and who loses from this scenario. Conclude with 2-3 concrete policy recommendations: should the government intervene further, adjust fiscal/monetary stance, invest in infrastructure, or coordinate with regional partners? Mention risks if the scenario is not managed properly.

=== WRITING STANDARDS ===
- Write in formal, technical economic prose suitable for policy briefs
- Use precise economic terminology: elasticities, multipliers, equilibrium adjustments, welfare effects, deadweight loss, rent-seeking, fiscal incidence
- Support claims with references to the provided data or contextual evidence
- Maintain objectivity; distinguish positive analysis (what will happen) from normative recommendations (what should be done)
- Target length: 400-500 words total across all five sections
- Audience: Policy analysts, trade ministry officials, development economists
- Output format: Plain text only—absolutely no markdown, asterisks, hashes, or special formatting

Begin your analysis now:"""


# Re-export for convenience
__all__ = [
    'merge_rag_content',
    'encode_query_to_vector',
    'query_rag_embeddings_semantic',
    'build_scenario_prompt'
]