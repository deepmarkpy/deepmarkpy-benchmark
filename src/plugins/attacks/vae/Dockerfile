FROM python:3.11-slim-buster AS builder

WORKDIR /app

# Set ARG for build-time use defaults
ARG APP_PORT=10001
ENV APP_PORT=${VAE_PORT}

ARG HOST="0.0.0.0"
ENV HOST=${HOST}

# Install dependencies in a temporary layer
COPY /src/plugins/attacks/vae/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final image
FROM python:3.11-slim-buster

WORKDIR /app

# Copy only the installed dependencies
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install system dependencies
RUN apt-get update

COPY src/utils ./utils

COPY src/plugins/attacks/vae .

# Expose the service port
EXPOSE 10001

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT