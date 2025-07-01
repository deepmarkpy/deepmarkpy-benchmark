FROM ml-services-base:latest

# Set ARG for build-time use defaults
ARG APP_PORT=5001
ENV APP_PORT=${AUDIOSEAL_PORT}

ARG HOST="0.0.0.0"
ENV HOST=${HOST}

# Only install service-specific dependencies not in base
COPY /src/plugins/models/audio_seal/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/plugins/models/audio_seal .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT