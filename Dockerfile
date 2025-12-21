FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY shared/ ./shared/
COPY trader/ ./trader/
COPY registry/ ./registry/

# Set Python path
ENV PYTHONPATH=/app
ENV MODE=paper

EXPOSE 8080

CMD ["uvicorn", "trader.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
