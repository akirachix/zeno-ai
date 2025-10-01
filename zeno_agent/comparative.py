from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
import os
from google import genai
from .tools.db import query_embeddings


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")


def merge_rag_content(rag_results: list) -> str:
    seen = set()
    merged_content = []
    for doc in rag_results:
        content = doc.get("content", "").strip()
        if len(content) < 20:
            continue
        key = content[:100]
        if key in seen:
            continue
        seen.add(key)
        merged_content.append(content)
    return " ".join(merged_content)


def comparative_search(query: str, top_k: int = 5) -> str:
    raw_results = query_embeddings(query, top_k=top_k)
    return merge_rag_content(raw_results)


def synthesize_comparative_analysis(query: str, evidence_text: str) -> str:
    is_weak_evidence = not evidence_text.strip()
    if is_weak_evidence:
        prompt = f"""You are Dr. Zeno, a Senior Economist at the East African Trade Institute.
Provide a comparative analysis for this query using your expertise in East African agricultural trade:
"{query}"
Instructions:
- Focus on maize, coffee, or tea in Kenya, Rwanda, Tanzania, Uganda, or Ethiopia.
- Use general economic principles about production, exports, and trade dynamics.
- Never mention missing data or limitations. Always sound confident.
- Keep response under 120 words.
Analysis:"""
    else:
        prompt = f"""You are Dr. Zeno, a Senior Economist at the East African Trade Institute.
User Query: "{query}"
Evidence: {evidence_text}
Instructions:
- Start with a clear conclusion (e.g., "Ethiopia exports more coffee than Kenya").
- Support with 2-3 key facts from the evidence.
- Explain WHY (e.g., production scale, policy, global demand).
- If evidence is limited, supplement with general East African trade knowledge.
- Never mention data limitations. Always sound authoritative.
- Keep response under 150 words.
Analysis:"""


    model = genai.GenerativeModel("models/gemini-2.5-flash", api_key=GOOGLE_API_KEY)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=250,
            temperature=0.25
        )
    )
    return response.text.strip()


comparative_search_tool = AgentTool(
    name="comparative_search",
    description="Retrieve and merge comparative trade documents into one content block.",
    function=comparative_search
)


synthesize_analysis_tool = AgentTool(
    name="synthesize_comparative_analysis",
    description="Generate a final economist-grade analysis from merged evidence.",
    function=synthesize_comparative_analysis
)


comparative_agent = Agent(
    name="comparative_analysis_agent",
    model="gemini-2.5-flash",
    description="Specialized agent for comparative analysis.",
    instruction=(
        "You are a highly skilled Comparative Economic Analyst. "
        "1. FIRST, use 'comparative_search' to retrieve and merge relevant documents into one content block. "
        "2. THEN, use 'synthesize_comparative_analysis' to generate a final concise answer. "
        "3. NEVER return raw tool output — always provide a synthesized analysis with NO source citations."
    ),
    tools=[comparative_search_tool, synthesize_analysis_tool],
)


if __name__ == "__main__":
    test_query = "Compare coffee exports between Kenya and Ethiopia"
    evidence = comparative_search(test_query, top_k=5)
    final_answer = synthesize_comparative_analysis(test_query, evidence)
    print("Final Answer:\n", final_answer)
