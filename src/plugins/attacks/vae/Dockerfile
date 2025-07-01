FROM ml-services-base:latest

# Set ARG for build-time use defaults
ARG APP_PORT=10001
ENV APP_PORT=${VAE_PORT}

ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/plugins/attacks/vae/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/plugins/attacks/vae .

# Expose the service port
EXPOSE 10001

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT