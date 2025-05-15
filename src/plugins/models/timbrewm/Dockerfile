FROM python:3.11-slim-buster AS builder

WORKDIR /app

# Set ARG for build-time use defaults
ARG APP_PORT=9001
ENV APP_PORT=${TIMBREWM_PORT}

ARG HOST="0.0.0.0"
ENV HOST=${HOST}

# Install dependencies in a temporary layer
COPY /src/plugins/models/timbrewm/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

RUN git clone https://github.com/TimbreWatermarking/TimbreWatermarking.git

# Rename the directory structure inside the repo
RUN if [ -d "TimbreWatermarking/watermarking_model/model" ]; then \
    mv TimbreWatermarking/watermarking_model/model TimbreWatermarking/watermarking_model/wm_model; \
    fi

# Final image
FROM python:3.11-slim-buster

WORKDIR /app

# Copy only the installed dependencies
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install system dependencies
RUN apt-get update

COPY src/utils ./utils

COPY src/plugins/models/timbrewm .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT