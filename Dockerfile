
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p data/raw data/uploaded data/vector_db

# Set Python path to include src directory
ENV PYTHONPATH=/app:/app/src
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run both services using the startup script
CMD ["python", "src/finance_insight_lite/app.py"]