"""
Main Zeno Agent API Server with FastAPI.

FILE: zeno_agent/agent.py
PURPOSE: FastAPI server with routing, streaming, and agent orchestration
"""

import os
import json
import asyncio
import traceback 
import re
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from google import genai
from typing import Dict, Any, AsyncGenerator

# Import agents
from zeno_agent.agents.comparative.comparative_agent import ComparativeAgent 
from zeno_agent.agents.forecasting.forecasting_agent import ForecastingAgent
from zeno_agent.agents.scenario.scenario_agent import ScenarioSubAgent
from zeno_agent.rag_tools import ask_knowledgebase

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")

# Google GenAI client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# -----------------------------------------------
# --- CONTENT CLEANING UTILITIES ---
# -----------------------------------------------

# Routing tags that should be filtered out from final responses
ROUTING_TAGS = [
    "[COMPARATIVE]", "[FORECAST]", "[SCENARIO]", "[RAG]",
    "[comparative]", "[forecast]", "[scenario]", "[rag]"
]

def clean_routing_tags(text: str) -> str:
    """Remove routing classification tags from text."""
    if not text:
        return text
    
    cleaned = text
    for tag in ROUTING_TAGS:
        cleaned = cleaned.replace(tag, "")
    
    # Also remove any remaining bracket patterns like [ANYTHING]
    cleaned = re.sub(r'\[(?:COMPARATIVE|FORECAST|SCENARIO|RAG)\]', '', cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip()

def is_routing_content(text: str) -> bool:
    """Check if text is just routing classification (should be filtered)."""
    if not text or not text.strip():
        return False
    
    cleaned = text.strip().upper()
    # If the entire content is just a routing tag, it's routing content
    return cleaned in ["[COMPARATIVE]", "[FORECAST]", "[SCENARIO]", "[RAG]"]

# -----------------------------------------------
# --- THREAD-SAFE ASYNC ITERATOR HELPER ---
# -----------------------------------------------

SENTINEL = object()

def stream_to_queue_sync(sync_generator, queue):
    """Convert synchronous generator to async queue."""
    try:
        for chunk in sync_generator:
            queue.put_nowait(chunk)
    except Exception as e:
        queue.put_nowait(e)
    finally:
        queue.put_nowait(SENTINEL)

async def aiter_from_queue(queue):
    """Async iterator from queue."""
    while True:
        item = await queue.get()
        if item is SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item
        queue.task_done()


# -----------------------------------------------
# --- ROUTING FUNCTION ---
# -----------------------------------------------
async def route_and_reason_stream(user_query: str) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream the routing LLM's reasoning using a robust, thread-safe queue pattern.
    Uses strict prompt with few-shot examples for accurate classification.
    """
    prompt = f"""You are Zeno, an AI Economist Assistant.

Classify the user's query into exactly ONE category.

Rules:
- Respond with ONLY this format: `[CATEGORY]`
- Valid categories: [COMPARATIVE], [FORECAST], [SCENARIO], [RAG]
- If it's a greeting, small talk, or general question, use [RAG]

Examples:
- "Compare coffee from Kenya and Ethiopia" → [COMPARATIVE]
- "Predict tea prices next year" → [FORECAST]
- "What if Rwanda subsidized tea production?" → [SCENARIO]
- "Hi", "How are you?", "Explain coffee exports" → [RAG]

User Query: "{user_query}"

Classification:"""
    
    def get_sync_stream():
        return client.models.generate_content_stream(
            model="gemini-2.0-flash", 
            contents=prompt,
        )
    
    queue = asyncio.Queue()
    
    try:
        stream_generator = await asyncio.to_thread(get_sync_stream)
        loop = asyncio.get_event_loop()
        producer_task = loop.run_in_executor(
            None, 
            stream_to_queue_sync, 
            stream_generator, 
            queue
        )

        full_text = ""
        async for chunk in aiter_from_queue(queue):
            text_chunk = getattr(chunk, 'text', '').strip()
            if text_chunk:
                full_text += text_chunk
                # Don't yield routing tags as thinking content
                if not is_routing_content(text_chunk):
                    yield {"type": "thinking", "content": text_chunk}

        await producer_task

        # Robust route detection
        normalized = full_text.strip().lower()

        route = None
        # Priority: exact bracket match first
        if "[comparative]" in normalized:
            route = "comparative"
        elif "[forecast]" in normalized:
            route = "forecast"
        elif "[scenario]" in normalized:
            route = "scenario"
        elif "[rag]" in normalized:
            route = "rag"
        # Fallback: keyword-in-text (only if no conflicting terms)
        elif "comparative" in normalized and not any(k in normalized for k in ["forecast", "scenario", "rag", "predict", "what if"]):
            route = "comparative"
        elif "forecast" in normalized or "predict" in normalized:
            route = "forecast"
        elif "scenario" in normalized or "what if" in normalized:
            route = "scenario"
        elif any(g in normalized for g in ["hi", "hello", "hey", "thanks", "ok", "help", "explain", "?"]):
            route = "rag"
        else:
            route = "rag"  # Safe default

        yield {"type": "route", "decision": route}

    except Exception as e:
        error_msg = f"Routing failed: {e}"
        yield {"type": "error", "message": error_msg}


# -----------------------------------------------
# --- ENTITY EXTRACTION HELPER ---
# -----------------------------------------------
def extract_entities(query: str) -> Dict[str, Any]:
    """Extract countries and commodities from query."""
    country_names = ["kenya", "tanzania", "uganda", "rwanda", "ethiopia", "burundi", "south sudan"]
    commodity_names = ["coffee", "maize", "tea", "beans", "wheat", "rice", "sugar", "oil", "sorghum"]
    query_lower = query.lower()
    countries = [c.title() for c in country_names if c in query_lower]
    commodities = [c for c in commodity_names if c in query_lower]
    return {
        "countries": countries[:2],
        "commodity": commodities[0] if commodities else "" 
    }


# -----------------------------------------------
# --- AGENT STREAMING WRAPPERS ---
# -----------------------------------------------
async def stream_scenario(query: str, file_context: str = "") -> AsyncGenerator[Dict[str, Any], None]:
    """Stream scenario analysis results."""
    yield {"type": "progress", "message": "Extracting entities and gathering context..."}
    await asyncio.sleep(0.1)

    queue = asyncio.Queue()
    try:
        agent = ScenarioSubAgent() 
        agent_stream_generator = await asyncio.to_thread(agent.stream_analysis, query, file_context)
        loop = asyncio.get_event_loop()
        producer_task = loop.run_in_executor(None, stream_to_queue_sync, agent_stream_generator, queue)
        
        async for event in aiter_from_queue(queue):
            yield event
            if event.get("status") == "SETUP_COMPLETE":
                yield {"type": "progress", "message": "Generating economic analysis..."}
            if event.get("type") == "error":
                break
        await producer_task
    except Exception as e:
        yield {"type": "error", "message": f"Scenario analysis failed: {e}"}


async def stream_comparative(query: str, file_context: str = "") -> AsyncGenerator[Dict[str, Any], None]:
    """Stream comparative analysis results."""
    yield {"type": "progress", "message": "Extracting countries and commodity..."}
    await asyncio.sleep(0.1)
    queue = asyncio.Queue()
    try:
        entities = extract_entities(query)
        if not entities["countries"] or not entities["commodity"]:
            yield {"type": "final", "response": "Please specify two countries and a commodity (e.g., coffee, maize)."}
            return
        
        yield {"type": "progress", "message": f"Fetching trade data for {entities['commodity']}..."}
        await asyncio.sleep(0.1)
        inputs = {"query": query, "file_context": file_context}
        agent_stream_generator = await asyncio.to_thread(ComparativeAgent.stream, inputs)
        loop = asyncio.get_event_loop()
        producer_task = loop.run_in_executor(None, stream_to_queue_sync, agent_stream_generator, queue)
        
        async for event in aiter_from_queue(queue):
            yield event
            if event.get("status") == "SETUP_COMPLETE":
                yield {"type": "progress", "message": "Generating final comparative analysis..."}
        await producer_task
    except Exception as e:
        yield {"type": "error", "message": f"Comparative analysis failed: {e}"}


async def stream_forecast(query: str, file_context: str = "") -> AsyncGenerator[Dict[str, Any], None]:
    """Stream forecast analysis results."""
    yield {"type": "progress", "message": "Preparing forecast model..."}
    await asyncio.sleep(0.5)
    try:
        forecasting_agent = ForecastingAgent()
        result = await asyncio.to_thread(forecasting_agent.run, {"query": query, "file_context": file_context})
        yield {"type": "final", "response": result.get("interpretation", result.get("response", "Forecast ready."))}
    except Exception as e:
        yield {"type": "error", "message": f"Forecast failed: {e}"}


# -----------------------------------------------
# --- MAIN QUERY ENDPOINT ---
# -----------------------------------------------
@app.post("/query")
async def query(request: Request):
    """Main endpoint handling user queries and streaming responses."""
    try:
        data = await request.json()
        user_query = data.get("query", "").strip()
        file_context = data.get("file_context", "").strip()

        if not user_query and not file_context:
            return StreamingResponse(
                iter([json.dumps({"error": "Query or file is required"}) + "\n"]),
                media_type="application/json"
            )

        async def generate():
            yield json.dumps({"type": "progress", "message": "Analyzing your question..."}) + "\n"
            await asyncio.sleep(0.2)

            route_decision = None
            trivial_response = None

            # Step 1: Stream routing reasoning
            async for event in route_and_reason_stream(user_query):
                # Don't send routing tags to frontend
                if event["type"] == "thinking" and is_routing_content(event.get("content", "")):
                    continue
                
                yield json.dumps(event) + "\n"
                
                if event["type"] == "route":
                    route_decision = event["decision"]
                elif event["type"] == "final":
                    trivial_response = event["response"]
                    route_decision = "trivial"
                elif event["type"] == "error":
                    return

            if route_decision == "trivial" and trivial_response:
                yield json.dumps({"type": "final", "response": trivial_response}) + "\n"
                return

            # Step 2: Handle routed agent
            if route_decision == "comparative":
                yield json.dumps({"type": "progress", "message": "Comparing commodities and trade metrics..."}) + "\n"
                await asyncio.sleep(0.2)
                
                full_comparative_response = ""
                
                async for event in stream_comparative(user_query, file_context):
                    if event["type"] == "error":
                        yield json.dumps(event) + "\n"
                        return
                    elif event["type"] == "progress":
                        yield json.dumps(event) + "\n"
                    elif event.get("type") == "comparative_chunk":
                        content = event.get("content", "")
                        if not is_routing_content(content):
                            cleaned_content = clean_routing_tags(content)
                            if cleaned_content:
                                full_comparative_response += cleaned_content
                                yield json.dumps({"type": "thinking", "content": cleaned_content}) + "\n"
                    elif event.get("type") == "comparative_complete":
                        break
                    else:
                        yield json.dumps(event) + "\n"

                if full_comparative_response:
                    final_response = clean_routing_tags(full_comparative_response.strip())
                    if final_response and len(final_response) > 20:
                        yield json.dumps({"type": "final", "response": final_response}) + "\n"
                    else:
                        yield json.dumps({"type": "error", "message": "Analysis generated no valid content. Please try rephrasing your question."}) + "\n"
                return

            elif route_decision == "forecast":
                yield json.dumps({"type": "progress", "message": "Accessing forecasting models..."}) + "\n"
                await asyncio.sleep(0.2)
                async for event in stream_forecast(user_query, file_context):
                    if event.get("type") == "final" and "response" in event:
                        event["response"] = clean_routing_tags(event["response"])
                    yield json.dumps(event) + "\n"
                    if event.get("type") in ("final", "error"):
                        return

            elif route_decision == "scenario":
                yield json.dumps({"type": "progress", "message": "Accessing scenario simulation engine..."}) + "\n"
                await asyncio.sleep(0.2)
                full_scenario_response = ""
                async for event in stream_scenario(user_query, file_context):
                    if event["type"] == "error":
                        yield json.dumps(event) + "\n"
                        return
                    elif event["type"] == "progress":
                        yield json.dumps(event) + "\n"
                    elif event.get("type") == "scenario_chunk":
                        content = event.get("content", "")
                        if not is_routing_content(content):
                            cleaned_content = clean_routing_tags(content)
                            if cleaned_content:
                                full_scenario_response += cleaned_content
                                yield json.dumps({"type": "thinking", "content": cleaned_content}) + "\n"
                    elif event.get("type") == "scenario_complete":
                        break
                    else:
                        yield json.dumps(event) + "\n"
                
                if full_scenario_response:
                    final_response = clean_routing_tags(full_scenario_response.strip())
                    if final_response and len(final_response) > 20:
                        yield json.dumps({"type": "final", "response": final_response}) + "\n"
                    else:
                        yield json.dumps({"type": "error", "message": "Analysis generated no valid content. Please try rephrasing your question."}) + "\n"
                return

            else:  # RAG
                yield json.dumps({"type": "progress", "message": "Retrieving knowledge..."}) + "\n"
                await asyncio.sleep(0.2)
                response = await asyncio.to_thread(ask_knowledgebase, user_query)
                final = response[0]["content"] if response else "No relevant information found."
                if file_context:
                    final = f"Uploaded document excerpt:\n{file_context[:500]}...\n\n{final}"
                final = clean_routing_tags(final)
                yield json.dumps({"type": "final", "response": final}) + "\n"
                return

        return StreamingResponse(generate(), media_type="application/json")

    except Exception as e:
        print(f"FATAL SERVER ERROR: {traceback.format_exc()}")
        return StreamingResponse(
            iter([json.dumps({"type": "error", "message": f"Server error: {e}"}) + "\n"]),
            media_type="application/json"
        )


# -----------------------------------------------
# --- HEALTH CHECK ENDPOINT ---
# -----------------------------------------------
@app.get("/healthz")
def health():
    """Health check endpoint."""
    return {"status": "ok"}