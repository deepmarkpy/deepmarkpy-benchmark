FROM python:3.10-slim AS builder

WORKDIR /app

# Install dependencies in a temporary layer
COPY /src/plugins/attacks/speech_tokenization/requirements.txt .
COPY /src/plugins/attacks/speech_tokenization/xcodec_requirements.txt .
RUN pip install --upgrade pip


#this is a package lobotomy, horrible!! 

RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install --no-cache-dir --user xcodec2==0.1.3 --no-deps
RUN pip install --no-cache-dir --user -r xcodec_requirements.txt

# Final image
FROM python:3.10-slim

WORKDIR /app

# Copy only the installed dependencies
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY src/utils ./utils

COPY src/plugins/attacks/speech_tokenization .

# Expose the service port
EXPOSE 10003

# Run the FastAPI server using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10003"]