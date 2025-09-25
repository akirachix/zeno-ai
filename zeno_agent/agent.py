import os
import re
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google.adk.agents import Agent as ADKAgent
from google.adk.tools.agent_tool import AgentTool
from tools.db import get_trade_data, semantic_search_rag_embeddings
from tools.graphing import plot_price_scenario
from scenario import ScenarioSubAgent
from comparative import comparative_agent
from forecasting import ForecastingAgent
from rag_tools import ask_knowledgebase

SUPPORTED_COUNTRIES = {"kenya", "rwanda", "tanzania", "uganda", "ethiopia"}
SUPPORTED_COMMODITIES = {"maize", "coffee", "tea"}

load_dotenv()

def load_prompt(filename: str) -> str:
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    with open(os.path.join(prompts_dir, filename), encoding="utf-8") as f:
        return f.read().strip()

def is_in_scope(query: str) -> bool:
    q = query.lower()
    return (any(c in q for c in SUPPORTED_COUNTRIES) and
            any(prod in q for prod in SUPPORTED_COMMODITIES))

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
                updated_str = str(updated_val)[:10] if updated_val else None
                source_str = f"{source_val}, updated {updated_str}" if updated_str else f"{source_val}" if source_val else "Unknown source"
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

forecasting_agent = ForecastingAgent()

def forecast_trade(
    commodity: str,
    metric: str,
    timeframe: str,
    country: str,
    model_type: Optional[str] = None,
    conversation_id: Optional[int] = None,
    run_id: Optional[int] = None
) -> Dict[str, Any]:
    if not all([commodity, metric, timeframe, country]):
        return {"error": "Missing required parameters: commodity, metric, timeframe, country"}

    commodity = commodity.lower().strip()
    metric = metric.lower().replace(" ", "_")
    country = country.lower().strip()

    commodity_mapping = {
        "coffee": "coffee_arabica" if country == "kenya" else "coffee_robusta",
        "tea": "tea"
    }
    normalized_commodity = commodity_mapping.get(commodity, commodity)

    if not re.match(r"next \d+ (years|months)", timeframe.lower()):
        return {"error": "Invalid timeframe format. Use 'next X years' or 'next X months'"}

    params = {
        "commodity": normalized_commodity,
        "metric": metric,
        "timeframe": timeframe,
        "country": country,
        "model_type": model_type,
        "conversation_id": conversation_id,
        "run_id": run_id,
        "original_commodity": commodity
    }

    try:
        result = forecasting_agent.run(params)
        if "error" in result:
            return {"error": result["error"]}

        forecast_value = result.get("forecast_value", "Unknown")
        confidence = result.get("confidence", "Medium")
        reasoning = result.get("reasoning", "")
        sources = result.get("sources", [])

        explanation = (
            f"Based on analysis of recent reports:\n"
            f" **Forecast**: {forecast_value}\n"
            f" **Confidence**: {confidence}\n"
            f" **Insight**: {reasoning}\n"
            f" **Sources**: {', '.join(sources[:3])}{'...' if len(sources) > 3 else ''}"
        )

        return {
            "result": result,
            "explanation": explanation
        }
    except Exception as e:
        return {"error": f"Forecasting failed: {str(e)}"}

comparative_tool = AgentTool(comparative_agent)
comparative_agent_raw = comparative_agent
scenario_agent_tool = scenario_tool
forecasting_tool = forecast_trade
rag_tool = ask_knowledgebase

root_agent = ADKAgent(
    name="zeno_root_agent",
    model="gemini-1.5-flash",
    description="The Zeno Root Agent is an AI Economist Assistant specialized in Kenyan Agricultural trade dynamics.",
    instruction="You are Zeno, an AI Economist Assistant for Kenya Agricultural Trade. Answer questions with economic rigor. Use tools for data tasks: forecast_trade, scenario_tool, comparative_tool, ask_knowledgebase. Always explain assumptions, cite sources, and offer to visualize or dive deeper.",
    tools=[
        forecasting_tool,
        scenario_agent_tool,
        comparative_tool,
        rag_tool,
    ],
)

app = FastAPI()

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    user_input = data.get("query", "")
    try:
        response = root_agent.run(user_input, tool_call_config={"allowed_tools": "any"})
        final_response = getattr(response, "text", response)
        return JSONResponse({"response": final_response})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/healthz")
def health():
    return {"status": "ok"}

def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("CRITICAL ERROR: GOOGLE_API_KEY environment variable is not set.")
        return

    print("Zeno Root Agent ready. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            break
        try:
            response = root_agent.run(user_input, tool_call_config={"allowed_tools": "any"})
            final_response = getattr(response, "text", response)
            print("Zeno:", final_response)
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    if os.environ.get("CLI_MODE", "0") == "1":
        main()
    else:
        import uvicorn
        port = int(os.environ.get("PORT", 8080))
        uvicorn.run("agent:app", host="0.0.0.0", port=port)
