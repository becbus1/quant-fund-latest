FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared module (now inside trader/ build context)
COPY shared/ ./shared/

# Copy trader module (app/ and __init__.py from build context)
COPY __init__.py ./trader/__init__.py
COPY app/ ./trader/app/

# Copy registry (now inside trader/ build context)
COPY registry/ ./registry/

# Set Python path
ENV PYTHONPATH=/app
ENV MODE=paper

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["uvicorn", "trader.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
