from fastapi import FastAPI
from .web_api import app as custom_app

def create_app() -> FastAPI:
    """ADK-compatible app factory."""
    app = FastAPI(
        title="Zeno Multi-Tool Agent API",
        description="Custom API endpoints...",
        version="1.0.0"
    )
    app.mount("/", custom_app)
    print("✅ Custom API mounted at root")
    return app

app = create_app()