FROM ml-services-base:latest

# Set ARG for build-time use defaults
ARG APP_PORT=10003
ENV APP_PORT=${SPEECH_TOKENIZATION_PORT}

ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/plugins/attacks/speech_tokenization/requirements.txt .
COPY /src/plugins/attacks/speech_tokenization/xcodec_requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir xcodec2==0.1.3 --no-deps && \
    pip install --no-cache-dir -r xcodec_requirements.txt

COPY src/plugins/attacks/speech_tokenization .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT