# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r zeno_agent/requirements.txt

EXPOSE 8000

CMD ["adk", "web"]
