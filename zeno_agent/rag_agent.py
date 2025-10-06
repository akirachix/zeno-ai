import os
from zeno_agent.tools.db import semantic_search_rag_embeddings

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "rag_agent_prompt.txt")

def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def synthesize_rag_answer(user_query: str, top_k=5) -> str:
    rag_results = semantic_search_rag_embeddings(user_query, top_k=top_k)
    context_str = "\n".join(
        f"- [{doc.get('source', 'Unknown')}] {doc.get('content', '')}" for doc in rag_results
    )
    prompt_template = load_prompt_template(PROMPT_PATH)
    prompt = prompt_template.format(
        user_query=user_query,
        context_str=context_str
    )
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": 500,
            "temperature": 0.2,
        }
    )
    return response.text.strip()