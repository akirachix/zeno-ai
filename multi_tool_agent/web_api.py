from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from agent import RootAgent
import os
import shutil

app = FastAPI()
agent = RootAgent()

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

@app.post("/scenario")
async def scenario(query: str = Form(...)):
    result = agent.route_query(query)
    graph_url = None

    graph_path = result.get("graph_path")
    if graph_path and os.path.exists(graph_path):
        filename = os.path.basename(graph_path)
        static_path = os.path.join(STATIC_DIR, filename)
        if os.path.abspath(graph_path) != os.path.abspath(static_path):
            shutil.copyfile(graph_path, static_path)
        graph_url = f"/static/{filename}"

    return JSONResponse({
        "response": result["response"],
        "graph_url": graph_url,
        "followup": result.get("followup", "")
    })

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")