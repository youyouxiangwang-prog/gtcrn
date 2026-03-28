FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY gtcrn.py .
COPY checkpoints/ ./checkpoints/
COPY test_wavs/ ./test_wavs/

# Copy API server
COPY api_server.py .

# Environment variables
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8005/health')" || exit 1

# Run the API server
CMD ["python", "api_server.py"]
