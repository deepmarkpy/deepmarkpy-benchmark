FROM ml-services-base:latest

# Set ARG for build-time use defaults
ARG APP_PORT=7010
ENV APP_PORT=${PERTH_PORT}

ARG HOST="0.0.0.0"
ENV HOST=${HOST}

# Only install service-specific dependencies not in base
COPY /src/plugins/models/perth/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/plugins/models/perth .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT