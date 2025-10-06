import os
import re
import json
import traceback
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import google.generativeai as genai
from zeno_agent.tools.db import get_trade_data, semantic_search_rag_embeddings, query_embeddings
from zeno_agent.tools.graphing import plot_price_scenario
from zeno_agent.scenario import ScenarioSubAgent
from zeno_agent.forecasting import ForecastingAgent
from zeno_agent.rag_tools import ask_knowledgebase
from zeno_agent.comparative import run_comparative_analysis

SUPPORTED_COUNTRIES = {"kenya", "rwanda", "tanzania", "uganda", "ethiopia"}
SUPPORTED_COMMODITIES = {"maize", "coffee", "tea"}

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def load_prompt_template(filename):
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_static_intro_message():
    return load_prompt_template("about_prompt.txt").strip()

def safe_gemini_text(response):
    try:
        if hasattr(response, "text") and response.text:
            return response.text
        if hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "content") and candidate.content.parts:
                    return "".join([str(p.text) for p in candidate.content.parts if hasattr(p, "text")])
    except Exception:
        pass
    return "Sorry, I couldn't generate a response for that query."

def synthesize_llm_response(prompt_file: str, **kwargs) -> str:
    prompt_template = load_prompt_template(prompt_file)
    prompt = prompt_template.format(**kwargs)
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 500, "temperature": 0.2}
    )
    return safe_gemini_text(response)

def clean_and_deduplicate_rag_results(rag_results: List[Dict]) -> List[Dict]:
    if isinstance(rag_results, dict) and rag_results.get("status") == "success":
        rag_results = rag_results.get("results", [])
    elif isinstance(rag_results, dict):
        return []
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
        cleaned.append({"content": content, "source": doc.get("source", "Unknown")})
    return cleaned

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

def extract_timeframe(q: str) -> Optional[str]:
    q = q.replace("yesrs", "years").replace("yr", "year")
    m = re.search(r"next\s+(\d+)\s+(years?|months?)", q, re.IGNORECASE)
    if m:
        num = m.group(1)
        unit = m.group(2).lower()
        return f"next {num} {unit}"
    return None

def extract_metric(q: str) -> Optional[str]:
    q_lower = q.lower()
    if "price" in q_lower:
        return "price"
    elif "export" in q_lower or "volume" in q_lower:
        return "export_volume"
    elif "revenue" in q_lower:
        return "revenue"
    return "export_volume"

def fuzzy_match_commodity(query: str) -> str:
    query_lower = query.lower()
    if "cofee" in query_lower or "coffe" in query_lower:
        return "coffee"
    elif "maiz" in query_lower:
        return "maize"
    elif "tea" in query_lower:
        return "tea"
    return None

def fuzzy_match_country(query: str) -> str:
    query_lower = query.lower()
    country_mapping = {
        "kenya": ["kenya", "kenyan"],
        "ethiopia": ["ethiopia", "ethiopian"],
        "uganda": ["uganda", "ugandan"],
        "tanzania": ["tanzania", "tanzanian"],
        "rwanda": ["rwanda", "rwandan"]
    }
    for country, variants in country_mapping.items():
        if any(variant in query_lower for variant in variants):
            return country
    return None

def scenario_tool(user_query: str) -> dict:
    thought_process = []
    commodity, country = parse_query(user_query)
    if not commodity:
        commodity = fuzzy_match_commodity(user_query)
    if not country:
        country = fuzzy_match_country(user_query) or "kenya"
    if not commodity:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        general_prompt = f"You are Zeno, an AI economist assistant. Answer this question based on general economic knowledge: {user_query}"
        response_obj = model.generate_content(general_prompt, generation_config={"max_output_tokens": 300, "temperature": 0.3})
        response = safe_gemini_text(response_obj)
        return {
            "response": response,
            "followup": "Try a question like: 'Forecast maize exports for Kenya next year?'"
        }
    if commodity and country and ("trend" in user_query.lower() or "historical" in user_query.lower()):
        db_result = get_trade_data(commodity, country, last_n_months=12, return_raw=True)
        months = db_result.get("months", [])
        prices = db_result.get("prices", [])
        if months and prices and len(prices) > 0:
            graph_path = plot_price_scenario(
                commodity, country, months, prices, prices, direction="none", pct=0
            )
            summary = (
                f"Here's the recent price trend for {commodity} in {country} (last 12 months).\n"
                f"Min price: {min(prices):.2f}, Max price: {max(prices):.2f}."
            )
            return {
                "response": summary + f"\n\n[Graph generated: {graph_path}]",
                "graph_path": graph_path,
                "thought_process": thought_process
            }
        else:
            articles = semantic_search_rag_embeddings(f"recent {commodity} price trends in {country}", top_k=3)
            if articles:
                summary = summarize_articles(articles)
                response = synthesize_llm_response(
                    "root_agent_prompt.txt",
                    user_query=f"What are the recent {commodity} price trends in {country}?",
                    context=summary
                )
                return {
                    "response": response,
                    "thought_process": thought_process
                }
            else:
                genai.configure(api_key=GOOGLE_API_KEY)
                model = genai.GenerativeModel("models/gemini-2.5-flash")
                general_prompt = f"You are Zeno, an AI economist assistant. Answer this question based on general economic knowledge: {user_query}"
                response_obj = model.generate_content(general_prompt, generation_config={"max_output_tokens": 300, "temperature": 0.3})
                response = safe_gemini_text(response_obj)
                return {
                    "response": response
                }
    articles = semantic_search_rag_embeddings(user_query, top_k=3)
    if articles:
        summary = summarize_articles(articles)
        response = synthesize_llm_response(
            "root_agent_prompt.txt",
            user_query=user_query,
            context=summary
        )
        return {
            "response": response,
            "thought_process": thought_process
        }
    result = ScenarioSubAgent().handle(user_query)
    if not result or not result.get("response"):
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        general_prompt = f"You are Zeno, an AI economist assistant. Answer this question based on general economic knowledge: {user_query}"
        response_obj = model.generate_content(general_prompt, generation_config={"max_output_tokens": 300, "temperature": 0.3})
        response = safe_gemini_text(response_obj)
        return {
            "response": response,
            "followup": "Try changing the country or commodity, or ask about Kenya, Rwanda, Tanzania, Uganda, or Ethiopia with maize, coffee, or tea."
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
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        general_prompt = f"You are Zeno, an AI economist assistant. Answer this question based on general economic knowledge: {commodity} {metric} {timeframe} {country}"
        response_obj = model.generate_content(general_prompt, generation_config={"max_output_tokens": 300, "temperature": 0.3})
        response = safe_gemini_text(response_obj)
        return {"error": response}
    commodity = commodity.lower().strip()
    metric = metric.lower().replace(" ", "_")
    country = country.lower().strip()
    if commodity in ["cofee", "coffe"]:
        commodity = "coffee"
    elif commodity in ["maiz"]:
        commodity = "maize"
    normalized_commodity = "coffee" if commodity == "coffee" else commodity
    if not re.match(r"next\s+\d+\s+(years?|months?)", timeframe.lower()):
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        general_prompt = f"You are Zeno, an AI economist assistant. Answer this question based on general economic knowledge: {commodity} {metric} {timeframe} {country}"
        response_obj = model.generate_content(general_prompt, generation_config={"max_output_tokens": 300, "temperature": 0.3})
        response = safe_gemini_text(response_obj)
        return {"error": response}
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
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            general_prompt = f"You are Zeno, an AI economist assistant. Answer this question based on general economic knowledge: {commodity} {metric} {timeframe} {country}"
            response_obj = model.generate_content(general_prompt, generation_config={"max_output_tokens": 300, "temperature": 0.3})
            response = safe_gemini_text(response_obj)
            return {"error": response}
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
            "response": explanation,
        }
    except Exception as e:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        general_prompt = f"You are Zeno, an AI economist assistant. Answer this question based on general economic knowledge: {commodity} {metric} {timeframe} {country}"
        response_obj = model.generate_content(general_prompt, generation_config={"max_output_tokens": 300, "temperature": 0.3})
        response = safe_gemini_text(response_obj)
        return {"error": response}

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
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(
            full_prompt,
            generation_config={"max_output_tokens": 400, "temperature": 0.2}
        )
        raw_text = safe_gemini_text(response).strip()
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON from model")
    except Exception as e:
        q = user_query.lower().strip()
        intro_phrases = [
            "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening",
            "who are you", "what is zeno", "tell me about yourself", "what can you do",
            "help", "about you", "your capabilities", "describe yourself", "introduce yourself"
        ]
        if any(q.startswith(phrase) for phrase in intro_phrases) or q in ["hello", "hi", "hey"]:
            return {"type": "who"}
        has_commodity = any(prod in q for prod in SUPPORTED_COMMODITIES) or "cofee" in q or "coffe" in q or "maiz" in q
        if any(kw in q for kw in ["what if", "scenario", "drop", "increase", "decrease", "shock", "impact of"]) and has_commodity:
            return {"type": "scenario"}
        if has_commodity and (extract_timeframe(user_query) or any(kw in q for kw in ["forecast", "predict", "project", "trend"])):
            return {"type": "forecast"}
        if any(kw in q for kw in ["compare", "vs", "versus", "difference", "relative", "between"]) and has_commodity:
            return {"type": "comparative"}
        return {"type": "rag"}

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
        if query_type == "who":
            return JSONResponse({"response": get_static_intro_message()})
        if query_type == "scenario":
            result = scenario_tool(user_query)
            return JSONResponse(result)
        elif query_type == "forecast":
            commodity = routed.get("commodity")
            if not commodity:
                commodity = fuzzy_match_commodity(user_query)
            detected_countries = [c for c in SUPPORTED_COUNTRIES if c in user_query.lower()]
            country = routed.get("country")
            if country:
                country = country.lower().strip()
            if not country:
                if detected_countries:
                    country = detected_countries[0]
                else:
                    country = fuzzy_match_country(user_query) or "kenya"
            country = country.lower().strip()
            metric = routed.get("metric") or extract_metric(user_query)
            timeframe = routed.get("timeframe") or extract_timeframe(user_query) or "next 2 years"
            params = {
                "commodity": commodity,
                "metric": metric,
                "timeframe": timeframe,
                "country": country,
            }
            if not params["commodity"]:
                genai.configure(api_key=GOOGLE_API_KEY)
                model = genai.GenerativeModel("models/gemini-2.5-flash")
                general_prompt = f"You are Zeno, an AI economist assistant. Answer this question based on general economic knowledge: {user_query}"
                response_obj = model.generate_content(general_prompt, generation_config={"max_output_tokens": 300, "temperature": 0.3})
                response = safe_gemini_text(response_obj)
                return JSONResponse({"response": response})
            result = forecast_trade(**params)
            return JSONResponse(result)
        elif query_type == "comparative":
            try:
                response_text = run_comparative_analysis(user_query)
                return JSONResponse({"response": response_text})
            except Exception as e:
                q_lower = user_query.lower()
                detected_countries = [c for c in SUPPORTED_COUNTRIES if c in q_lower]
                detected_commodities = [c for c in SUPPORTED_COMMODITIES if c in q_lower]
                if not detected_commodities:
                    fuzzy_commodity = fuzzy_match_commodity(user_query)
                    if fuzzy_commodity:
                        detected_commodities = [fuzzy_commodity]
                if not detected_commodities:
                    genai.configure(api_key=GOOGLE_API_KEY)
                    model = genai.GenerativeModel("models/gemini-2.5-flash")
                    general_prompt = f"You are Zeno, an AI economist assistant. Answer this question based on general economic knowledge: {user_query}"
                    response_obj = model.generate_content(general_prompt, generation_config={"max_output_tokens": 300, "temperature": 0.3})
                    response = safe_gemini_text(response_obj)
                    return JSONResponse({"response": response})
                if len(detected_countries) == 0:
                    detected_countries = ["kenya", "uganda"]
                elif len(detected_countries) == 1:
                    other_countries = [c for c in SUPPORTED_COUNTRIES if c != detected_countries[0]]
                    if other_countries:
                        detected_countries.append(other_countries[0])
                raw_rag = query_embeddings(user_query, top_k=5)
                rag_results = clean_and_deduplicate_rag_results(raw_rag)
                if not rag_results:
                    specific_query = f"compare {detected_commodities[0]} exports between {' and '.join(detected_countries[:2])}"
                    raw_rag = query_embeddings(specific_query, top_k=5)
                    rag_results = clean_and_deduplicate_rag_results(raw_rag)
                evidence_blocks = []
                for doc in rag_results:
                    content = doc["content"]
                    if len(content) > 300:
                        content = content[:300].rsplit(" ", 1)[0] + "..."
                    evidence_blocks.append(content)
                evidence_text = " ".join(evidence_blocks) if evidence_blocks else "No specific evidence found."
                comparative_prompt = load_prompt_template("comparative_agent_prompt.txt")
                prompt = comparative_prompt.format(user_query=user_query, evidence_text=evidence_text)
                genai.configure(api_key=GOOGLE_API_KEY)
                model = genai.GenerativeModel("models/gemini-2.5-flash")
                response = model.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": 400, "temperature": 0.2}
                )
                return JSONResponse({"response": safe_gemini_text(response)})
        else:
            rag_results = ask_knowledgebase(user_query)
            if rag_results:
                context = "\n".join(
                    f"- [{doc.get('source', 'Unknown')}] {doc.get('content', '')}" for doc in rag_results
                )
                response = synthesize_llm_response(
                    "root_agent_prompt.txt",
                    user_query=user_query,
                    context=context
                )
            else:
                genai.configure(api_key=GOOGLE_API_KEY)
                model = genai.GenerativeModel("models/gemini-2.5-flash")
                general_prompt = f"""
                You are Zeno, an AI economist assistant. Answer the following question clearly and professionally based on general economic knowledge.

                Question: {user_query}

                Provide a concise, accurate response. If the question is about your capabilities, explain that you specialize in East African agricultural trade but can answer general economic questions.
                """
                response_obj = model.generate_content(
                    general_prompt,
                    generation_config={"max_output_tokens": 300, "temperature": 0.3}
                )
                response = safe_gemini_text(response_obj)
            return JSONResponse({"response": response})
    except Exception as e:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        general_prompt = f"You are Zeno, an AI economist assistant. Answer this question based on general economic knowledge: {user_query}"
        response_obj = model.generate_content(general_prompt, generation_config={"max_output_tokens": 300, "temperature": 0.3})
        response = safe_gemini_text(response_obj)
        return JSONResponse({"response": response}, status_code=500)

@app.get("/healthz")
def health():
    return {"status": "ok"}