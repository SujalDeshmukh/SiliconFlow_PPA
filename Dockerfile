FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set Python path so imports work
ENV PYTHONPATH=/app/src

# Environment config (can be overridden in docker-compose)
ENV PPA_SCENARIO=small_4block
ENV PPA_MAX_STEPS=50
ENV PPA_RANDOMISE=false

EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "envs.silicon_flow_ppa.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
