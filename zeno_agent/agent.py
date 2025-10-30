import os
import time
import hashlib
from typing import Dict, Any, Optional, Tuple
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv

load_dotenv()

from zeno_agent.agents.comparative.comparative_agent import comparative_agent
from zeno_agent.agents.forecasting.forecasting_agent import ForecastingAgent
from zeno_agent.agents.scenario.scenario_agent import ScenarioSubAgent
from zeno_agent.rag_tools import ask_knowledgebase

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not set.")

CACHE_TTL_SECONDS = 600  
_cache: Dict[str, Tuple[float, Any]] = {}

def _make_cache_key(prefix: str, query: str) -> str:
    h = hashlib.sha256(query.lower().encode("utf-8")).hexdigest()[:16]
    return f"{prefix}::{h}"

def cache_get(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    if not entry:
        return None
    ts, value = entry
    if time.time() - ts > CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return value

def cache_set(key: str, value: Any) -> None:
    _cache[key] = (time.time(), value)

def lightweight_route(user_query: str) -> Dict[str, Any]:
    q = user_query.lower().strip()
    if not q:
        return {"type": "trivial", "response": "Hello! How can I help with East African trade data?"}

    trivial_stopwords = {"hello", "hi", "hey", "thanks", "thank", "date", "time"}

    if len(q.split()) <= 4 and any(w in q for w in trivial_stopwords):
        return {"type": "trivial", "response": "Hello! I specialize in East African agricultural trade. How can I help?"}

    if any(k in q for k in ["compare", "vs", "versus", "difference", "between", "compared"]):
        return {"type": "comparative", "response": ""}

    if any(k in q for k in ["forecast", "predict", "projection", "trend", "next", "quarter"]):
        return {"type": "forecast", "response": ""}

    if any(k in q for k in ["what if", "what happens","scenario", "increase", "decrease", "drop", "shock", "if"]):
        return {"type": "scenario", "response": ""}

    return {"type": "rag", "response": ""}

def route_and_reason(user_query: str) -> Dict[str, Any]:
    return lightweight_route(user_query)

def detect_output_format(query: str) -> Dict[str, bool]:
    q = query.lower()
    return {
        "include_chart": any(word in q for word in ["chart", "graph", "plot", "visual", "figure", "show.*trend"]),
        "include_csv": any(word in q for word in ["csv", "download", "export", "spreadsheet", "data", "table", "dataset"]),
        "include_excel": any(word in q for word in ["excel", "xlsx", "sheet"])
    }

app = FastAPI()

async def handle_user_query(user_query: str) -> Dict[str, Any]:
    start = time.time()
    routed = route_and_reason(user_query)
    routing_time = time.time() - start

    qtype = routed.get("type", "rag")
    trivial_response = routed.get("response", "")

    if qtype == "trivial" and trivial_response:
        return {
            "type": "trivial",
            "answer": trivial_response,
            "context": None,
            "sources": [],
            "timings": {"routing": routing_time, "total": time.time() - start}
        }

    cache_key = _make_cache_key(qtype, user_query)
    cached = cache_get(cache_key)
    if cached:
        cached["timings"] = {"routing": routing_time, "cached": True, "total": time.time() - start}
        return cached

    output_format = detect_output_format(user_query)

    if qtype == "comparative":
        t0 = time.time()
        try:
            result = await run_in_threadpool(comparative_agent.run, {"query": user_query})
        except TypeError:
            result = await run_in_threadpool(comparative_agent.run, user_query)
        elapsed = time.time() - t0
        response = {
            "type": "comparative",
            "answer": result.get("analysis") or result.get("response") or str(result),
            "data": result,
            "sources": result.get("sources", []),
            "timings": {"routing": routing_time, "execution": elapsed, "total": time.time() - start}
        }
        cache_set(cache_key, response)
        return response

    elif qtype == "forecast":
        t0 = time.time()
        forecasting_agent = ForecastingAgent()
        result = await run_in_threadpool(forecasting_agent.run, {"query": user_query})
        elapsed = time.time() - t0

        if "error" in result:
            return {
                "type": "forecast",
                "answer": result["error"],
                "data": {"error": result["error"]},
                "sources": [],
                "timings": {"routing": routing_time, "execution": elapsed, "total": time.time() - start}
            }

        human_answer = result.get("response", "Forecast generated.")
        
        response = {
            "type": "forecast",
            "answer": human_answer,
            "data": result, 
            "sources": result.get("sources", []),
            "timings": {"routing": routing_time, "execution": elapsed, "total": time.time() - start}
        }
        cache_set(cache_key, response)
        return response

    elif qtype == "scenario":
        t0 = time.time()
        result = await run_in_threadpool(ScenarioSubAgent().handle, user_query)
        elapsed = time.time() - t0
        response = {
            "type": "scenario",
            "answer": result.get("response") or result.get("explanation") or str(result),
            "data": result,
            "sources": [result.get("source")] if result.get("source") else [],
            "timings": {"routing": routing_time, "execution": elapsed, "total": time.time() - start}
        }
        cache_set(cache_key, response)
        return response

    else: 
        t0 = time.time()
        rag_text = await run_in_threadpool(ask_knowledgebase, user_query)
        elapsed = time.time() - t0
        response = {
            "type": "rag",
            "answer": rag_text,
            "data": {"response": rag_text},
            "sources": [],
            "timings": {"routing": routing_time, "execution": elapsed, "total": time.time() - start}
        }
        cache_set(cache_key, response)
        return response

@app.post("/query")
async def query_endpoint(request: Request):
    payload = await request.json()
    q = payload.get("query", "")
    if not isinstance(q, str):
        q = str(q) if q is not None else ""
    q = q.strip()
    if not q:
        return JSONResponse({"error": "Query is required"}, status_code=400)
    try:
        result = await handle_user_query(q)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/healthz")
def health():
    return {"status": "ok"}