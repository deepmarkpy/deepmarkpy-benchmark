FROM ml-services-base:latest AS builder

# Set ARG for build-time use defaults
ARG APP_PORT=10004
ENV APP_PORT=${NEURAL_VOCODER_PORT}

ARG HOST="0.0.0.0"
ENV HOST=${HOST}

# Handle BigVGAN in builder stage
RUN git clone https://github.com/NVIDIA/BigVGAN.git 
COPY /src/plugins/attacks/neural_vocoder/big_vgan_requirements.txt BigVGAN/requirements.txt
RUN pip install --no-cache-dir -r BigVGAN/requirements.txt

FROM ml-services-base:latest

# Copy BigVGAN from builder
COPY --from=builder /app/BigVGAN /app/BigVGAN

COPY src/utils /app/BigVGAN/app_utils
COPY src/plugins/attacks/neural_vocoder/config.json /app/BigVGAN/config.json

COPY /src/plugins/attacks/neural_vocoder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/plugins/attacks/neural_vocoder .

WORKDIR /app/BigVGAN

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT