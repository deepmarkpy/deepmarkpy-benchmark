FROM python:3.10-slim-buster AS builder

WORKDIR /app

# Set ARG for build-time use defaults
ARG APP_PORT=9001
ENV APP_PORT=${TIMBREWM_PORT}

ARG HOST="0.0.0.0"
ENV HOST=${HOST}

RUN apt-get update && \
    apt-get install -y git gcc libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies in a temporary layer
COPY /src/plugins/models/timbrewm/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

RUN git clone https://github.com/TimbreWatermarking/TimbreWatermarking.git

# Rename the directory structure inside the repo
RUN if [ -d "TimbreWatermarking/watermarking_model/model" ]; then \
    mv TimbreWatermarking/watermarking_model/model TimbreWatermarking/watermarking_model/wm_model; \
    fi

# Fix import in mel_transform.py
RUN sed -i 's/from frequency import window_sumsquare/from .frequency import window_sumsquare/g' TimbreWatermarking/watermarking_model/distortions/mel_transform.py
# Remove torchaudio import from conv2_mel_modules.py
RUN sed -i '/import torchaudio/d' TimbreWatermarking/watermarking_model/wm_model/conv2_mel_modules.py
# Fix path in hifigan config
RUN sed -i 's|hifigan/config.json|TimbreWatermarking/watermarking_model/hifigan/config.json|g' TimbreWatermarking/watermarking_model/wm_model/conv2_mel_modules.py
# Fix device handling in hifigan model loading
RUN sed -i 's|ckpt = torch.load("./hifigan/model/VCTK_V1/generator_v1")|gpu_available = torch.cuda.is_available()\n    device = "cuda" if gpu_available else "cpu"\n    ckpt = torch.load("./hifigan/model/VCTK_V1/generator_v1", map_location=torch.device(device))|g' TimbreWatermarking/watermarking_model/wm_model/conv2_mel_modules.py
# Fix path in hifigan model loading
RUN sed -i 's|./hifigan/model/VCTK_V1/generator_v1|TimbreWatermarking/watermarking_model/hifigan/model/VCTK_V1/generator_v1|g' TimbreWatermarking/watermarking_model/wm_model/conv2_mel_modules.py


# Final image
FROM python:3.10-slim-buster

WORKDIR /app

# Copy only the installed dependencies
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY --from=builder /app/TimbreWatermarking /app/TimbreWatermarking

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY src/utils ./utils

COPY src/plugins/models/timbrewm .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT