import os
import re
import json
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import google.generativeai as genai
from zeno_agent.tools.db import get_trade_data, semantic_search_rag_embeddings
from zeno_agent.tools.graphing import plot_price_scenario
from zeno_agent.scenario import ScenarioSubAgent
from zeno_agent.forecasting import ForecastingAgent

SUPPORTED_COUNTRIES = {"kenya", "rwanda", "tanzania", "uganda", "ethiopia"}
SUPPORTED_COMMODITIES = {"maize", "coffee", "tea"}

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
genai.configure(api_key=GOOGLE_API_KEY)

def merge_rag_content(rag_results: list) -> str:
    """Merge all RAG content into one cohesive text block (NO sources, deduplicated)."""
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

def is_in_scope(query: str) -> bool:
    q = query.lower()
    return (any(c in q for c in SUPPORTED_COUNTRIES) and
            any(prod in q for prod in SUPPORTED_COMMODITIES))

def parse_query(query: str):
    q = query.lower()
    commodity = next((prod for prod in SUPPORTED_COMMODITIES if prod in q), None)
    country = next((cty for cty in SUPPORTED_COUNTRIES if cty in q), None)
    return commodity, country

def scenario_tool(user_query: str) -> dict:
    thought_process = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    thought_process.append(f"[Time] {now_str}")
    thought_process.append(f"[Step 1] Checking if query is in supported countries/commodities: {user_query}")
    if not is_in_scope(user_query):
        return {
            "response": (
                "I specialize in East African agricultural trade dynamics for maize, coffee, and tea across Kenya, Rwanda, Tanzania, Uganda, and Ethiopia. "
                "For other topics, I recommend consulting regional agricultural authorities or trade ministries for the most current insights."
            ),
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
            articles = semantic_search_rag_embeddings(user_query, top_k=3)
            if articles:
                merged_content = merge_rag_content(articles)
                prompt = f"""You are Dr. Zeno, Senior Economist at East African Trade Institute.
User Query: "{user_query}"
Context: {merged_content}
Instructions: Provide a concise analysis of {commodity} price trends in {country} using available context. Never mention data limitations. Keep under 100 words."""
                model = genai.GenerativeModel("models/gemini-pro")
                response = model.generate_content(prompt)
                return {
                    "response": response.text.strip(),
                    "thought_process": thought_process
                }
            else:
                prompt = f"""You are Dr. Zeno, Senior Economist at East African Trade Institute.
Provide authoritative analysis of {commodity} price trends in {country} based on East African agricultural economics principles. Keep under 100 words."""
                model = genai.GenerativeModel("models/gemini-pro")
                response = model.generate_content(prompt)
                return {
                    "response": response.text.strip(),
                    "thought_process": thought_process
                }

    articles = semantic_search_rag_embeddings(user_query, top_k=3)
    if articles:
        merged_content = merge_rag_content(articles)
        prompt = f"""You are Dr. Zeno, Senior Economist at East African Trade Institute.
User Query: "{user_query}"
Context: {merged_content}
Instructions: Provide a concise scenario analysis using available context. Never mention data limitations. Keep under 120 words."""
        model = genai.GenerativeModel("models/gemini-pro")
        response = model.generate_content(prompt)
        return {
            "response": response.text.strip(),
            "thought_process": thought_process
        }
    else:
        prompt = f"""You are Dr. Zeno, Senior Economist at East African Trade Institute.
Provide authoritative scenario analysis for "{user_query}" based on East African agricultural economics principles. Keep under 120 words."""
        model = genai.GenerativeModel("models/gemini-pro")
        response = model.generate_content(prompt)
        return {
            "response": response.text.strip(),
            "thought_process": thought_process
        }

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
        model = genai.GenerativeModel("models/gemini-1.5-flash")
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
            evidence_text = merge_rag_content(raw_rag)

            prompt = f"""You are Dr. Zeno, a Senior Economist at the East African Trade Institute. 
User Query: "{user_query}"
Evidence: {evidence_text}
Instructions:
- Start with a clear conclusion (e.g., "Ethiopia exports more coffee than Kenya").
- Support with 2-3 key facts from the evidence.
- Explain WHY (e.g., production scale, policy, global demand).
- If evidence is limited, supplement with general East African trade knowledge.
- Never mention data limitations. Always sound confident and authoritative.
- Keep it under 150 words.
Analysis:"""

            model = genai.GenerativeModel("models/gemini-pro")
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=250,
                    temperature=0.2
                )
            )
            return JSONResponse({"response": response.text.strip()})

        else:
            from zeno_agent.tools.db import semantic_search_rag_embeddings
            articles = semantic_search_rag_embeddings(user_query, top_k=5)
            
            if articles:
                merged_content = merge_rag_content(articles)
                prompt = f"""You are Dr. Zeno, a Senior Economist at the East African Trade Institute. 
User Query: "{user_query}"
Available Context: {merged_content}
Instructions:
- Provide a concise, evidence-based response using the available context.
- If context is limited, supplement with general economic knowledge about East African agricultural trade.
- Never mention missing data or limitations. Always sound confident and authoritative.
- Keep response under 120 words.
Response:"""
                model = genai.GenerativeModel("models/gemini-pro")
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=200,
                        temperature=0.3
                    )
                )
                return JSONResponse({"response": response.text.strip()})
            else:
                prompt = f"""You are Dr. Zeno, a Senior Economist at the East African Trade Institute. 
Provide a concise, authoritative response to this query about East African agricultural trade:
"{user_query}"
Instructions:
- Use your expertise in East African economics to provide a reasoned response.
- Focus on maize, coffee, or tea in Kenya, Rwanda, Tanzania, Uganda, or Ethiopia.
- Never admit knowledge gaps. Always sound confident and professional.
- Keep response under 100 words.
Response:"""
                model = genai.GenerativeModel("models/gemini-pro")
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=150,
                        temperature=0.4
                    )
                )
                return JSONResponse({"response": response.text.strip()})

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
                    evidence_text = merge_rag_content(raw_rag)
                    prompt = f"""You are Dr. Zeno, a Senior Economist at the East African Trade Institute. 
User Query: "{user_input}"
Evidence: {evidence_text}
Instructions: Start with a clear conclusion. Support with 2-3 facts. Explain drivers. No source citations. Under 150 words. Only output analysis.
Analysis:"""
                    model = genai.GenerativeModel("models/gemini-pro")
                    response = model.generate_content(prompt)
                    print("Zeno:", response.text.strip())
            else:
                from zeno_agent.tools.db import semantic_search_rag_embeddings
                articles = semantic_search_rag_embeddings(user_input, top_k=5)
                if articles:
                    merged_content = merge_rag_content(articles)
                    prompt = f"""You are Dr. Zeno, Senior Economist. "{user_input}" Context: {merged_content}. Provide concise response under 120 words."""
                    model = genai.GenerativeModel("models/gemini-pro")
                    response = model.generate_content(prompt)
                    print("Zeno:", response.text.strip())
                else:
                    prompt = f"""You are Dr. Zeno, Senior Economist. Provide authoritative response to "{user_input}" under 100 words."""
                    model = genai.GenerativeModel("models/gemini-pro")
                    response = model.generate_content(prompt)
                    print("Zeno:", response.text.strip())
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