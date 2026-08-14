FROM python:3.11-slim

WORKDIR /app

COPY requirements-docker.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY phase4 ./phase4
COPY phase5 ./phase5

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "phase5.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
