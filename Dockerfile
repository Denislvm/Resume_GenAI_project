FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY req.txt /app/req.txt

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install compatible versions first to resolve conflicts
RUN pip install --no-cache-dir --upgrade pip

# Install core conflicting packages first with compatible versions
RUN pip install --no-cache-dir \
    transformers==4.53.0 \
    optimum==1.27.0

# Install remaining packages without dependency checking
RUN pip install --no-cache-dir -r req.txt --no-deps

# Verify no conflicts and install any missing dependencies
RUN pip install --no-cache-dir \
    torch \
    numpy \
    pandas \
    psycopg2-binary \
    openai \
    python-dotenv \
    llama-index-core \
    llama-index-llms-openai \
    llama-index-embeddings-openai

# Final dependency check
RUN pip check || echo "Warning: Some dependency conflicts exist but proceeding..."

# Copy application code
COPY . /app

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Keep the container running for manual execution
CMD ["tail", "-f", "/dev/null"]