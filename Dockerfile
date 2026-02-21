# syntax=docker/dockerfile:1

# ============================================================
# Stage 1: Builder - Build wheels and install dependencies
# ============================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE /build/
COPY src /build/src

# Install to user site-packages (easier to copy to final stage)
RUN pip install --user --no-cache-dir --upgrade pip \
 && pip install --user --no-cache-dir ".[api]"

# ============================================================
# Stage 2: Runtime - Minimal production image
# ============================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy source code
COPY src /app/src
COPY pyproject.toml README.md LICENSE /app/

# Add .local/bin to PATH
ENV PATH=/root/.local/bin:$PATH

# Cloud Run listens on $PORT
ENV PORT=8080
EXPOSE 8080

# Run the application
CMD ["sh", "-c", "python -m uvicorn student_performance.api:app --host 0.0.0.0 --port ${PORT:-8080}"]