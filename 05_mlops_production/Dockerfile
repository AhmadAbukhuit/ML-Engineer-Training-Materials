# Multi-stage lean Dockerfile for production FastAPI ML microservice
FROM python:3.11-slim as builder

WORKDIR /app

# Install build tools if necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final runtime image
FROM python:3.11-slim

WORKDIR /app

# Create unprivileged application user
RUN useradd -m -u 1000 appuser

# Copy installed wheels and dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application source and model weights
COPY 05_mlops_production/app /app/app
COPY 03_machine_learning/models /app/03_machine_learning/models

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
