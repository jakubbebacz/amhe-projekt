# Use official Python runtime
FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src

# Run experiments script with unbuffered output
ENTRYPOINT ["python", "-u", "src/cma_experiments.py"]