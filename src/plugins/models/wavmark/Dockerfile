FROM ml-services-base:latest

WORKDIR /app

# Set ARG for build-time use defaults
ARG APP_PORT=8001
ENV APP_PORT=${WAVMARK_PORT}

ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/plugins/models/wavmark/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/plugins/models/wavmark .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT