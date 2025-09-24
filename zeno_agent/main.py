# main.py
from fastapi import FastAPI
from .web_api import app as custom_app

def create_app() -> FastAPI:
    app = FastAPI(
        title="Zeno Agent API",
        description="...",
        version="1.0.0"
    )
    
    @app.get("/")
    def root():
        return {"status": "ok"}

    app.mount("/api", custom_app)
    return app

app = create_app()
