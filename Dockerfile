# Multi-stage build for Lead Scoring System

FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv and dependencies
RUN pip install uv fastapi uvicorn streamlit plotly

# Copy dependency files
COPY pyproject.toml uv.lock ./
RUN uv pip install --system -e . || pip install -e .

# Copy application code
COPY app/ ./app/
COPY dashboard/ ./dashboard/
COPY config.py ./
COPY models/ ./models/
COPY outputs/ ./outputs/
COPY data/processed/ ./data/processed/

ENV PYTHONPATH=/app

# FastAPI stage
FROM base as api
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Streamlit stage
FROM base as dashboard
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
