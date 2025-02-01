# Use an official lightweight Python image
FROM python:3.11-slim-buster

# Prevent Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1

# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1

# Install system dependencies required for Python extensions
RUN apt-get update \
    && apt-get install -y gcc libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies
COPY /src/plugins/models/silent_cipher/requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY src/utils ./utils

COPY src/plugins/models/silent_cipher .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7001"]