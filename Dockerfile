FROM python:3.11
WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Only if you need .env in your container (for python-dotenv)

EXPOSE 8000

# Use Cloud Run's port env var
CMD ["sh", "-c", "uvicorn multi_tool_agent.web_api:app --host 0.0.0.0 --port $PORT"]