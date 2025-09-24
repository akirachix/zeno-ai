from fastapi import FastAPI
from .web_api import app as custom_app

def create_app() -> FastAPI:
    """
    Create the FastAPI app with your custom endpoints mounted at root.
    This enables /docs, /openapi.json, and all your routes.
    """
    app = FastAPI(
        title="Zeno Agent API",
        description="Custom API for East African agri-trade analysis: scenario, forecast, comparative, and RAG tools.",
        version="1.0.0",
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    @app.get("/", summary="Health Check")
    def health_check():
        return {"status": "ok", "message": "Zeno Agent is running."}
    app.mount("/api", custom_app)

    return app
app = create_app()
