import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai
from zeno_agent.tools.db import get_trade_data, semantic_search_rag_embeddings
from zeno_agent.agents.forecasting import ForecastingAgent
from zeno_agent.agents.forecasting.forecasting_agent import ForecastingAgent
from zeno_agent.rag_tools import ask_knowledgebase
from zeno_agent.agents.comparative.comparative_agent import comparative_agent
from zeno_agent.agents.scenario.scenario_agent import ScenarioSubAgent

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")

client = genai.Client(api_key=GOOGLE_API_KEY)

def route_and_reason(user_query: str) -> dict:
    """
     Gemini LLM fully reason about the query.
    It decides:
    - type: trivial / comparative / forecast / scenario / rag
    - trivial queries are answered directly by LLM
    Returns dict {"type": ..., "response": ...}
    """
    prompt = f"""
       You are Zeno, an AI Economist Assistant specializing in East African agricultural trade data.

      Your task:
      1. Analyze the user's query.
      2. If it is a greeting, small talk, or simple factual question unrelated to trade data (e.g., "Hello", "What is the date today?"),
     answer it naturally.
       3. If it is about comparing countries/commodities → indicate [COMPARATIVE] at the start of your response.
         4. If it is asking for a forecast or prediction → indicate [FORECAST].
         5. If it is a hypothetical or what-if scenario → indicate [SCENARIO].
         6. If it requires document retrieval or knowledge lookup or understanding of trade atmosphere in Eastern Africa → indicate [RAG].

         Do not output JSON. Just respond naturally or with the tags above.

         Query: "{user_query}"
        """

    try:
        result = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        raw_output = result.text.strip()

        if raw_output.startswith("[COMPARATIVE]"):
            return {"type": "comparative", "response": ""}
        elif raw_output.startswith("[FORECAST]"):
            return {"type": "forecast", "response": ""}
        elif raw_output.startswith("[SCENARIO]"):
            return {"type": "scenario", "response": ""}
        elif raw_output.startswith("[RAG]"):
            return {"type": "rag", "response": ""}
        else:
            return {"type": "trivial", "response": raw_output}

    except Exception as e:
        print(f" LLM call failed: {e}")
        query = user_query.lower()
        trade_keywords = {"export", "import", "price", "trade", "coffee", "maize", "tanzania", "kenya", "forecast", "compare"}
        if len(query.split()) <= 6 and not any(key_word in query for key_word in trade_keywords):
            return {"type": "trivial", "response": "Hello! I specialize in East African agricultural trade data. How can I help?"}
        return {"type": "rag", "response": ""}

# FASTAPI APP
app = FastAPI()

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    user_query = data.get("query", "").strip()
    if not user_query:
        return JSONResponse({"error": "Query is required"}, status_code=400)

    try:
        routed = route_and_reason(user_query)
        query_type = routed.get("type", "rag")
        trivial_response = routed.get("response", "")

        if query_type == "trivial" and trivial_response:
            return JSONResponse({"type": "trivial", "response": trivial_response})

        elif query_type == "comparative":
             result = comparative_agent.run({"query": user_query})
             return JSONResponse(result)


        
        elif query_type == "forecast":
            forecasting_agent = ForecastingAgent()
            result = forecasting_agent.run({"query": user_query})
            return JSONResponse({"type": "forecast", **result})

        elif query_type == "scenario":
            result = ScenarioSubAgent().handle(user_query)
            return JSONResponse({"type": "scenario", **result})

        else:
            response = ask_knowledgebase(user_query)
            return JSONResponse({"type": "rag", "response": response})

    except Exception as e:
        return JSONResponse({"error": f"Processing failed: {str(e)}"}, status_code=500)

# HEALTH CHECK
@app.get("/healthz")
def health():
    return {"status": "ok"}
