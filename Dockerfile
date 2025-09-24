FROM python:3.11
WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r multi_tool_agent/requirements.txt

EXPOSE 8000

CMD ["sh", "-c", "uvicorn multi_tool_agent.web_api:app --host 0.0.0.0 --port ${PORT:-8000}"]