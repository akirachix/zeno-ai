import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai
import traceback

from zeno_agent.agents.comparative.comparative_agent import comparative_agent
from zeno_agent.agents.forecasting.forecasting_agent import ForecastingAgent
from zeno_agent.agents.scenario.scenario_agent import ScenarioSubAgent
from zeno_agent.rag_tools import ask_knowledgebase_with_context 

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
client = genai.Client(api_key=GOOGLE_API_KEY)

app = FastAPI()

def route_and_reason(user_query: str) -> dict:
    prompt = f"""
You are Zeno, an AI Economist Assistant specializing in East African agricultural trade data.

Your task:
1. Analyze the user's query.
2. If it is a greeting, small talk, or simple factual question unrelated to trade data (e.g., "Hello", "What is the date today?"),
answer it naturally.
3. If it is about comparing countries/commodities → indicate [COMPARATIVE] at the start of your response.
4. If it is asking for a forecast or prediction → indicate [FORECAST].
5. If it is a hypothetical or what-if scenario → indicate [SCENARIO].
6. If it requires document retrieval or knowledge lookup → indicate [RAG].

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
        print(f"LLM call failed: {e}")
        query = user_query.lower()
        trade_keywords = {"export", "import", "price", "trade", "coffee", "maize", "tanzania", "kenya", "forecast", "compare"}
        if len(query.split()) <= 6 and not any(key_word in query for key_word in trade_keywords):
            return {"type": "trivial", "response": "Hello! I specialize in East African agricultural trade data. How can I help?"}
        return {"type": "rag", "response": ""}


@app.post("/query")
async def query(request: Request):
    try:
        data = await request.json()
        user_query = data.get("query", "").strip()
        file_context = data.get("file_context", "").strip()

        if not user_query and not file_context:
            return JSONResponse({"error": "Query or file is required"}, status_code=400)

        if file_context and not user_query:

            prompt = f"""
You are Dr. Zeno, Senior Economist. A user uploaded a document but didn't ask a specific question.
Your primary role is to ensure the user gets value from their uploaded file.

Document Context (includes filename and content):
{file_context}

Instructions:
1. Provide a brief, professional 1-2 sentence summary of the main topic or data presented in the uploaded documents.
2. Suggest 3 specific, actionable economic questions based on the document content that an economist would be interested in.
"""
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                analysis = response.text.strip()
            except Exception:
                analysis = "I analyzed your document. Try asking about trade implications, price forecasts, or policy impacts."
            return JSONResponse({
                "type": "file_analysis",
                "response": analysis,
                "followup": "Ask one of the suggested questions!"
            })

        routed = route_and_reason(user_query)
        query_type = routed.get("type", "rag")

        if query_type == "trivial":
            return JSONResponse({"type": "trivial", "response": routed["response"]})

        elif query_type == "comparative":
            result = comparative_agent.run({
                "query": user_query,
                "file_context": file_context  
            })
            return JSONResponse(result)

        elif query_type == "forecast":
            forecasting_agent = ForecastingAgent()
            result = forecasting_agent.run({
                "query": user_query,
                "file_context": file_context  
            })
            return JSONResponse({"type": "forecast", **result})

        elif query_type == "scenario":
            result = ScenarioSubAgent().handle_with_context(user_query, file_context)
            return JSONResponse({"type": "scenario", **result})

        else:  

            base_response = ask_knowledgebase_with_context(user_query, file_context)
            return JSONResponse({"type": "rag", "response": base_response})

    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        print("ERROR in /query:", traceback.format_exc())
        return JSONResponse({"error": error_msg}, status_code=500)


@app.get("/healthz")
def health():
    return {"status": "ok"}