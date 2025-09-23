FROM python:3.11
WORKDIR /app

COPY . /app
COPY multi_tool_agent /app/agents/multi_tool_agent

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["sh", "-c", "uvicorn multi_tool_agent.web_api:app --host 0.0.0.0 --port $PORT"]
