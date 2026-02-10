FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional but useful for wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy project metadata first for caching
COPY pyproject.toml README.md LICENSE /app/

# Copy source
COPY src /app/src

# Install your package + deps
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir . \
 && pip install --no-cache-dir ".[api]"  # for serving the API; add other server deps if needed

# Copy artifacts
#COPY artifacts /app/artifacts

# Cloud Run listens on $PORT
ENV PORT=8080
EXPOSE 8080

# Example: if api.py is at src/student_performance/api.py then :
#   student_performance.api:app
CMD ["sh", "-c", "uvicorn student_performance.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
