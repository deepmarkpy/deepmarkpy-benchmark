FROM python:3.11-slim-buster AS builder

WORKDIR /app

# Install dependencies in a temporary layer
COPY /src/plugins/models/wavmark/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final image
FROM python:3.11-slim-buster

WORKDIR /app

# Copy only the installed dependencies
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install system dependencies
RUN apt-get update

COPY src/utils ./utils

COPY src/plugins/models/wavmark .

# Expose the service port
EXPOSE 8001

# Run the FastAPI server using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]