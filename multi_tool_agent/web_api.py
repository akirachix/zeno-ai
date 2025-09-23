from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from agent import scenario_tool, forecast_trade, rag_tool, comparative_agent_instance
import os
import shutil

app = FastAPI()

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

@app.get("/")
def health_check():
   return {"status": "ok", "message": "Multi Tool Agent is running."}

@app.post("/scenario")
async def scenario(query: str = Form(...)):
   result = scenario_tool(query)
   graph_url = None
   graph_path = result.get("graph_path")
   if graph_path and os.path.exists(graph_path):
       filename = os.path.basename(graph_path)
       static_path = os.path.join(STATIC_DIR, filename)
       if os.path.abspath(graph_path) != os.path.abspath(static_path):
           shutil.copyfile(graph_path, static_path)
       graph_url = f"/static/{filename}"

   return JSONResponse({
       "response": result.get("response", ""),
       "graph_url": graph_url,
       "followup": result.get("followup", ""),
       "thought_process": result.get("thought_process", [])
   })

@app.post("/forecast")
async def forecast(
   commodity: str = Form(...),
   metric: str = Form(...),
   timeframe: str = Form(...),
   country: str = Form(...),
   model_type: str = Form(None)
):
   result = forecast_trade(commodity, metric, timeframe, country, model_type=model_type)
   return JSONResponse(result)

@app.post("/comparative")
async def comparative(query: str = Form(...)):
   try:
       response = comparative_agent_instance.run(query)
       result_text = getattr(response, "text", str(response))
       return JSONResponse({"result": result_text})
   except Exception as e:
       return JSONResponse({"error": f"Comparative analysis failed: {str(e)}"}, status_code=500)

@app.post("/rag")
async def rag(query: str = Form(...)):
   result = rag_tool(query)
   return JSONResponse({"results": result})

@app.post("/run")
async def run(request: Request):
   data = await request.json()
   endpoint_type = data.get("type", "scenario")
   try:
       if endpoint_type == "scenario":
           result = scenario_tool(data.get("query", ""))
       elif endpoint_type == "forecast":
           result = forecast_trade(
               data.get("commodity", ""),
               data.get("metric", ""),
               data.get("timeframe", ""),
               data.get("country", ""),
               model_type=data.get("model_type")
           )
       elif endpoint_type == "comparative":
           response = comparative_agent_instance.run(data.get("query", ""))
           result_text = getattr(response, "text", str(response))
           result = {"result": result_text}
       elif endpoint_type == "rag":
           result = rag_tool(data.get("query", ""))
       else:
           result = {"error": f"Unknown type: {endpoint_type}"}
       return JSONResponse(result if isinstance(result, dict) else {"result": result})
   except Exception as e:
       return JSONResponse({"error": f"Endpoint '{endpoint_type}' failed: {str(e)}"}, status_code=500)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
