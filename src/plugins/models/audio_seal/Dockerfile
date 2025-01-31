FROM python:3.9-slim AS builder

WORKDIR /app

# Install dependencies in a temporary layer
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final image
FROM python:3.9-slim

WORKDIR /app

# Copy only the installed dependencies
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install system dependencies
RUN apt-get update

COPY . .

# Expose the service port
EXPOSE 5001

# Run the FastAPI server using Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5001"]