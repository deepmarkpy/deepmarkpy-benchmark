FROM python:3.11-slim-buster AS builder

WORKDIR /app

# Set ARG for build-time use defaults
ARG APP_PORT=10004
ENV APP_PORT=${NEURAL_VOCODER_PORT}

ARG HOST="0.0.0.0"
ENV HOST=${HOST}

# Install dependencies in a temporary layer
COPY /src/plugins/attacks/neural_vocoder/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN pip install --no-cache-dir --user --upgrade pip
RUN git clone https://github.com/NVIDIA/BigVGAN.git 
COPY /src/plugins/attacks/neural_vocoder/big_vgan_requirements.txt BigVGAN/requirements.txt
RUN pip install --no-cache-dir --user -r BigVGAN/requirements.txt

# Final image
FROM python:3.11-slim-buster

WORKDIR /app

# Copy only the installed dependencies
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy BigVGAN from builder
COPY --from=builder /app/BigVGAN /app/BigVGAN

# Install system dependencies
RUN apt-get update

COPY src/utils ./utils

COPY src/plugins/attacks/neural_vocoder .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT