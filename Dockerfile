FROM python:3.12-slim

WORKDIR /app

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

RUN pip install --no-cache-dir -r req.txt --no-deps

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

RUN pip check || echo "Warning: Some dependency conflicts exist but proceeding..."

COPY . /app

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["tail", "-f", "/dev/null"]