FROM ml-services-base:latest AS builder

ARG APP_PORT=9001
ENV APP_PORT=${APP_PORT}

ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/plugins/models/timbrewm/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/TimbreWatermarking/TimbreWatermarking.git

RUN cd TimbreWatermarking && \
    if [ -d "watermarking_model/model" ]; then \
        mv watermarking_model/model watermarking_model/wm_model; \
    fi && \
    sed -i 's/from frequency import window_sumsquare/from .frequency import window_sumsquare/g' \
        watermarking_model/distortions/mel_transform.py && \
    sed -i '/import torchaudio/d' \
        watermarking_model/wm_model/conv2_mel_modules.py && \
    sed -i 's|hifigan/config.json|TimbreWatermarking/watermarking_model/hifigan/config.json|g' \
        watermarking_model/wm_model/conv2_mel_modules.py && \
    sed -i 's|ckpt = torch.load("./hifigan/model/VCTK_V1/generator_v1")|gpu_available = torch.cuda.is_available()\n    device = "cuda" if gpu_available else "cpu"\n    ckpt = torch.load("./hifigan/model/VCTK_V1/generator_v1", map_location=torch.device(device))|g' \
        watermarking_model/wm_model/conv2_mel_modules.py && \
    sed -i 's|./hifigan/model/VCTK_V1/generator_v1|TimbreWatermarking/watermarking_model/hifigan/model/VCTK_V1/generator_v1|g' \
        watermarking_model/wm_model/conv2_mel_modules.py

FROM ml-services-base:latest

ARG APP_PORT=9001
ARG HOST="0.0.0.0"
ENV APP_PORT=${APP_PORT}
ENV HOST=${HOST}

COPY --from=builder /app/TimbreWatermarking /app/TimbreWatermarking

COPY /src/plugins/models/timbrewm/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/plugins/models/timbrewm .

EXPOSE ${APP_PORT}

CMD ["sh", "-c", "uvicorn app:app --host=$HOST --port=$APP_PORT"]