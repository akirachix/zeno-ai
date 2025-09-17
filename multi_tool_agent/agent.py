import os
from dotenv import load_dotenv
from datetime import datetime

from google.adk.agents.llm_agent import Agent
from .tools.db import get_trade_data, semantic_search_rag_embeddings
from .tools.graphing import plot_price_scenario
from .scenario import ScenarioSubAgent

SUPPORTED_COUNTRIES = {"kenya", "rwanda", "tanzania", "uganda", "ethiopia"}
SUPPORTED_COMMODITIES = {"maize", "coffee", "tea"}

def load_prompt(filename):
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    with open(os.path.join(prompts_dir, filename), encoding="utf-8") as f:
        return f.read().strip()

load_dotenv()

def is_in_scope(query: str) -> bool:
    q = query.lower()
    return (any(c in q for c in SUPPORTED_COUNTRIES)
            and any(prod in q for prod in SUPPORTED_COMMODITIES))

def parse_query(query: str):
    q = query.lower()
    commodity = next((prod for prod in SUPPORTED_COMMODITIES if prod in q), None)
    country = next((cty for cty in SUPPORTED_COUNTRIES if cty in q), None)
    return commodity, country

def summarize_articles(articles):
    if not articles:
        return "No relevant articles or reports found."
    summary_lines = []
    for idx, article in enumerate(articles, 1):
        context = article.get("content") or article.get("context") or article.get("text") or str(article)
        snippet = (context[:180] + "...") if len(context) > 180 else context
        summary_lines.append(f"- Article {idx}: {snippet}")
    return "\n".join(summary_lines)

def scenario_tool(user_query: str) -> dict:
    thought_process = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    thought_process.append(f"[Time] {now_str}")

    thought_process.append(f"[Step 1] Checking if query is in supported countries/commodities: {user_query}")
    if not is_in_scope(user_query):
        return {
            "response": (
                "Hey there! I’m Zeno, your friendly East African agri-trade advisor. "
                "Right now, I can only help with maize, coffee, or tea in Kenya, Rwanda, Tanzania, Uganda, or Ethiopia. "
                "If you need info outside these, just ask for help in this area! If I don't know something, I'll let you know honestly."
            ),
            "followup": "Try a question like: 'Forecast maize exports for Kenya next year?'",
            "thought_process": thought_process
        }

    commodity, country = parse_query(user_query)
    thought_process.append(f"[Step 2] Parsed commodity: {commodity}, country: {country}")
    if commodity and country and ("trend" in user_query.lower() or "historical" in user_query.lower()):
        db_result = get_trade_data(commodity, country, last_n_months=12, return_raw=True)
        months = db_result.get("months", [])
        prices = db_result.get("prices", [])
        meta = db_result.get("metadata")
        thought_process.append(f"[Step 3] Fetched months: {months}")
        thought_process.append(f"[Step 4] Fetched prices: {prices}")
        if months and prices:
            graph_path = plot_price_scenario(
                commodity, country, months, prices, prices, direction="none", pct=0
            )
            if meta:
                source_val = getattr(meta, "source", None) or meta.get("source")
                updated_val = getattr(meta, "updated_at", None) or meta.get("updated_at")
                if updated_val:
                    updated_str = str(updated_val)[:10]
                    source_str = f"{source_val}, updated {updated_str}"
                else:
                    source_str = f"{source_val}" if source_val else "Unknown source"
            else:
                source_str = "Unknown source"
            summary = (
                f"Here's the recent price trend for {commodity} in {country} (last 12 months).\n"
                f"Min price: {min(prices):.2f}, Max price: {max(prices):.2f}.\n"
                f"Source: {source_str}"
            )
            thought_process.append(f"[Step 5] Generated graph at {graph_path}")
            return {
                "response": summary + f"\n\n[Graph generated: {graph_path}]",
                "graph_path": graph_path,
                "thought_process": thought_process
            }
        else:
            thought_process.append("[Step 5] No data found in DB.")
            return {
                "response": f"No data found for {commodity} in {country} yet. Try a different combo, or upload more info—I'm ready for more!",
                "thought_process": thought_process
            }

    thought_process.append("[Step 6] Performing semantic search for unstructured scenario analysis from RAG embeddings...")
    articles = semantic_search_rag_embeddings(user_query, top_k=3)
    if articles:
        thought_process.append(f"[Step 7] Found {len(articles)} relevant articles or RAG-embedded data.")
        summary = summarize_articles(articles)
        thought_process.append("[Step 8] Synthesized evidence from RAG embeddings.")
        return {
            "response": (
                "Based on my analysis of relevant RAG-embedded articles and reports in the database, here's what they say:\n"
                + summary
            ),
            "thought_process": thought_process
        }

    thought_process.append("[Step 9] Delegating to ScenarioSubAgent for scenario simulation.")
    result = ScenarioSubAgent().handle(user_query)
    if not result or not result.get("response"):
        thought_process.append("[Step 10] ScenarioSubAgent could not generate a response.")
        return {
            "response": (
                "Looks like I don't have enough info for that yet. "
                "Try changing the country, commodity, or upload more data—I'm always learning!"
            ),
            "followup": "Try changing the country or commodity, or ask about Kenya, Rwanda, Tanzania, Uganda, or Ethiopia with maize, coffee, or tea.",
            "thought_process": thought_process
        }
    if result.get("graph_path"):
        result["response"] += f"\n\n[Graph generated: {result['graph_path']}]"
    result["thought_process"] = thought_process
    return result

root_agent = Agent(
    model="gemini-2.5-flash",
    name="zeno_agent",
    description="Zeno Root Agent: East African Agri-Trade Economic Analysis",
    instruction=(
        "Hello! I'm Zeno, your assistant for economic analysis of agri-trade in East Africa. "
        "I can answer questions and analyze scenarios about maize, coffee, or tea in Kenya, Rwanda, Tanzania, Uganda, or Ethiopia."
        "\n\n"
        "Please ask about these commodities and countries only. For example:"
        "\n- 'What is the impact of a 20% drop in maize price in Kenya over the next 3 months?'"
        "\n- 'How would a drought impact coffee production in Ethiopia?'"
        "\n- 'What are the recent price trends for tea in Uganda?'"
        "\n\n"
        "If you ask about other commodities or countries, I'll let you know I'm not equipped to answer yet but don't sound dumb just be cool and confident be a cool AI that agents will love also show users your thought process."
        "\n\n"
        "If I don't have enough data, I'll be honest and suggest what you can try next."
    ),
    tools=[scenario_tool],
)