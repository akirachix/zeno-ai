FROM python:3.11-slim

WORKDIR /app
COPY . /app

ENV PYTHONPATH=/app

RUN pip install --no-cache-dir -r zeno_agent/requirements.txt
RUN pip install fastapi uvicorn

EXPOSE 8080

CMD ["uvicorn", "zeno_agent.agent:app", "--host", "0.0.0.0", "--port", "8080"]
