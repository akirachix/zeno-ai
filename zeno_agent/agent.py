import os
import re
import json
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai
from zeno_agent.tools.db import get_trade_data, semantic_search_rag_embeddings
from zeno_agent.tools.graphing import plot_price_scenario
from zeno_agent.scenario import ScenarioSubAgent
from zeno_agent.forecasting import ForecastingAgent
from zeno_agent.rag_tools import ask_knowledgebase

SUPPORTED_COUNTRIES = {"kenya", "rwanda", "tanzania", "uganda", "ethiopia"}
SUPPORTED_COMMODITIES = {"maize", "coffee", "tea"}

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
genai.configure(api_key=GOOGLE_API_KEY)

def clean_and_deduplicate_rag_results(rag_results: list) -> list:
    seen = set()
    cleaned = []
    for doc in rag_results:
        content = doc.get("content", "").strip()
        if len(content) < 20:
            continue
        key = content[:100]
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"content": content})
    return cleaned

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
        summary_lines.append(f"- {snippet}")
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
        thought_process.append(f"[Step 3] Fetched months: {months}")
        thought_process.append(f"[Step 4] Fetched prices: {prices}")
        if months and prices:
            graph_path = plot_price_scenario(
                commodity, country, months, prices, prices, direction="none", pct=0
            )
            summary = (
                f"Here's the recent price trend for {commodity} in {country} (last 12 months).\n"
                f"Min price: {min(prices):.2f}, Max price: {max(prices):.2f}."
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
                "Based on my analysis of relevant reports in the database, here's what they say:\n"
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

        explanation = (
            f"Based on analysis of recent reports:\n"
            f" **Forecast**: {forecast_value}\n"
            f" **Confidence**: {confidence}\n"
            f" **Insight**: {reasoning}"
        )

        return {
            "result": result,
            "explanation": explanation
        }
    except Exception as e:
        return {"error": f"Forecasting failed: {str(e)}"}

ROUTER_PROMPT = """
You are Zeno, an AI Economist Assistant for East African agricultural trade.

Classify the user's query into one of these types:
- "scenario": for "what if", hypothetical shocks, price drops/increases, policy impacts.
- "forecast": for predictions about future values (price, export volume, revenue).
- "comparative": for comparisons between countries, crops, or time periods.
- "rag": for general knowledge questions not requiring data analysis.

Also extract key parameters when possible:
- commodity (maize, coffee, tea)
- country (Kenya, Uganda, etc.)
- metric (price, export_volume, revenue) — only for forecast
- percentage (e.g., 20) — only for scenario
- direction (increase/decrease) — only for scenario
- timeframe (e.g., "next 2 years") — for forecast/scenario

Respond ONLY in valid JSON format with this structure:
{{
  "type": "scenario|forecast|comparative|rag",
  "commodity": "...",
  "country": "...",
  "metric": "...",
  "percentage": 20,
  "direction": "decrease",
  "timeframe": "next 1 year"
}}

If a field is unknown, omit it or set to null.
User query: "{query}"
""".strip()

def route_query(user_query: str) -> dict:
    try:
        full_prompt = ROUTER_PROMPT.format(query=user_query)
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(full_prompt)
        raw_text = response.text.strip()
        
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()
        
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON from model")
            
    except Exception as e:
        q = user_query.lower()
        if user_query.lower().startswith("what is"):
            return {"type": "rag"}
        if any(w in q for w in ["what if", "scenario", "drop", "increase", "decrease", "shock", "fall", "rise"]):
            typ = "scenario"
        elif any(w in q for w in ["forecast", "predict", "next year", "next month", "next 2 years", "trend", "project"]):
            typ = "forecast"
        elif any(w in q for w in ["compare", "vs", "versus", "difference", "relative", "between .* and"]):
            typ = "comparative"
        else:
            typ = "rag"
        return {"type": typ}

app = FastAPI()

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    user_query = data.get("query", "").strip()
    
    if not user_query:
        return JSONResponse({"error": "Query is required"}, status_code=400)

    try:
        routed = route_query(user_query)
        query_type = routed.get("type", "rag")

        if query_type == "scenario":
            result = scenario_tool(user_query)
            return JSONResponse(result)

        elif query_type == "forecast":
            params = {
                "commodity": routed.get("commodity"),
                "metric": routed.get("metric", "price"),
                "timeframe": routed.get("timeframe", "next 2 years"),
                "country": routed.get("country"),
            }
            if not (params["commodity"] and params["country"]):
                return JSONResponse({
                    "error": "Could not extract commodity and country. Please specify them clearly (e.g., 'maize in Kenya')."
                })
            result = forecast_trade(**params)
            return JSONResponse(result)

        elif query_type == "comparative":
            q_lower = user_query.lower()
            detected_countries = [c for c in SUPPORTED_COUNTRIES if c in q_lower]
            detected_commodities = [c for c in SUPPORTED_COMMODITIES if c in q_lower]

            if len(detected_countries) < 2:
                return JSONResponse({
                    "error": "Please specify at least two countries to compare (e.g., 'Kenya and Ethiopia')."
                })
            if not detected_commodities:
                return JSONResponse({
                    "error": "Please specify a commodity (maize, coffee, or tea)."
                })

            from zeno_agent.tools.query import query_embeddings
            raw_rag = query_embeddings(user_query, top_k=5)
            rag_results = clean_and_deduplicate_rag_results(raw_rag)
            
            evidence_blocks = []
            for doc in rag_results:
                content = doc["content"]
                if len(content) > 300:
                    content = content[:300].rsplit(" ", 1)[0] + "..."
                evidence_blocks.append(content)
            
            evidence_text = " ".join(evidence_blocks) if evidence_blocks else "No specific evidence found."

            prompt = f"""You are Dr. Zeno, a Senior Economist at the East African Trade Institute. 
User Query: "{user_query}"
Evidence: {evidence_text}
Instructions:
- Start with a clear conclusion (e.g., "Ethiopia exports more coffee than Kenya").
- Support with 2-3 key facts from the evidence.
- Explain WHY (e.g., production scale, policy, global demand).
- Output ONLY the analysis — no disclaimers, no lists, no markdown, no source citations.
- Keep it under 150 words.
Analysis:"""

            model = genai.GenerativeModel("models/gemini-2.5-flash")
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=250,
                    temperature=0.2
                )
            )
            return JSONResponse({"response": response.text.strip()})

        else:
            response = ask_knowledgebase(user_query)
            return JSONResponse({"response": response})

    except Exception as e:
        return JSONResponse({"error": f"Processing failed: {str(e)}"}, status_code=500)

@app.get("/healthz")
def health():
    return {"status": "ok"}

import asyncio

async def run_cli():
    print("Zeno Agent (Routing Mode) ready. Type 'quit' to exit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        try:
            routed = route_query(user_input)
            print(f"[DEBUG] Routed to: {routed}")
            if routed["type"] == "scenario":
                result = scenario_tool(user_input)
                print("Zeno:", result["response"])
            elif routed["type"] == "forecast":
                params = {
                    "commodity": routed.get("commodity", "maize"),
                    "metric": routed.get("metric", "price"),
                    "timeframe": routed.get("timeframe", "next 2 years"),
                    "country": routed.get("country", "kenya"),
                }
                result = forecast_trade(**params)
                print("Zeno:", result.get("explanation", result))
            elif routed["type"] == "comparative":
                q_lower = user_input.lower()
                detected_countries = [c for c in SUPPORTED_COUNTRIES if c in q_lower]
                detected_commodities = [c for c in SUPPORTED_COMMODITIES if c in q_lower]
                if len(detected_countries) < 2 or not detected_commodities:
                    print("Zeno: Please specify two countries and a commodity (e.g., 'coffee in Kenya and Ethiopia').")
                else:
                    from zeno_agent.tools.query import query_embeddings
                    raw_rag = query_embeddings(user_input, top_k=5)
                    rag_results = clean_and_deduplicate_rag_results(raw_rag)
                    evidence_blocks = []
                    for doc in rag_results:
                        content = doc["content"]
                        if len(content) > 300:
                            content = content[:300].rsplit(" ", 1)[0] + "..."
                        evidence_blocks.append(content)
                    evidence_text = " ".join(evidence_blocks) if evidence_blocks else "No evidence found."
                    prompt = f"""You are Dr. Zeno, a Senior Economist at the East African Trade Institute. 
User Query: "{user_input}"
Evidence: {evidence_text}
Instructions: Start with a clear conclusion. Support with 2-3 facts. Explain drivers. No source citations. Under 150 words. Only output analysis.
Analysis:"""
                    model = genai.GenerativeModel("models/gemini-2.5-flash")
                    response = model.generate_content(prompt)
                    print("Zeno:", response.text.strip())
            else:
                print("Zeno:", ask_knowledgebase(user_input))
        except Exception as e:
            print(f"ERROR: {e}")

def main():
    asyncio.run(run_cli())

if __name__ == "__main__":
    if os.environ.get("CLI_MODE", "0") == "1":
        main()
    else:
        import uvicorn
        port = int(os.environ.get("PORT", 8080))
        uvicorn.run("agent:app", host="0.0.0.0", port=port)