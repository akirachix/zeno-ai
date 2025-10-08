from zeno_agent.embedding_utils import encode_query_to_vector
from zeno_agent.db_utils import query_rag_embeddings_semantic


def merge_rag_content(rag_results):
    seen = set()
    merged = []
    for doc in rag_results:
        content = doc.get("content", "").strip()
        if len(content) < 20:
            continue
        key = content[:100]
        if key in seen:
            continue
        seen.add(key)
        merged.append(content)
    return " ".join(merged)


def build_scenario_prompt(query, structured_context, rag_context):
   return f"""
You are Dr. Zeno, Senior Economist at the East African Trade Institute.
Write a detailed professional **plain text** economic analysis of the following policy or market scenario.


Scenario Query: "{query}"


Available Context:
- Structured Economic Data: {structured_context}
- Policy, Events, and Macroeconomic Documents: {rag_context}


Instructions:
- Use clear, formal economic language (elasticities, fiscal multipliers, trade balance, general/partial equilibrium).
- Organize the analysis into short, readable paragraphs (no Markdown or bullet points).
- Discuss:
  1. Immediate Effects
  2. Price Dynamics
  3. Trade Effects
  4. Macroeconomic Implications
  5. Risks and Recommendations
- Base reasoning on the data provided.
- Keep it between 250–300 words.
- Output must be plain text only — no bolding, asterisks, hashes, or list markers.
"""

