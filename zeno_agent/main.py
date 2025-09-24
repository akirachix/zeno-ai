from fastapi import FastAPI
from .web_api import app as custom_app

def create_app() -> FastAPI:
    app = FastAPI()
    app.mount("/", custom_app)
    print("✅ Mounted at /api")
    return app

app = create_app()
