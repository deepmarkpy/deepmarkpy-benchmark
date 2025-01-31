# Use a builder stage for installing dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy dependencies file
COPY requirements.txt .

# Install dependencies in a temporary layer
RUN pip install --no-cache-dir --user -r requirements.txt

# Final image
FROM python:3.9-slim

WORKDIR /app

# Copy only installed dependencies
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUSERBASE=/root/.local
ENV PYTHONPATH=/root/.local/lib/python3.9/site-packages:$PYTHONPATH

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY . .

# Expose the service port
EXPOSE 5001

# Run FastAPI server using Python module
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5001"]