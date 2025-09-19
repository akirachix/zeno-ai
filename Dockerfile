FROM python:3.11
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
CMD ["sh", "-c", "uvicorn web_api:app --host 0.0.0.0 --port $PORT"]
