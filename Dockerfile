#Dockerfile.train
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

#Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

#Set working directory
WORKDIR /app

#Copy dependency file
COPY requirements.txt .

#Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

#Copy project files
COPY . .

#Set environment variables
ENV PYTHONUNBUFFERED=1

#Default command
CMD ["python3", "scr/train.py"]

#Dockerfile
FROM python:3.10-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

#Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copy code and models
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

#Expose port
EXPOSE 8000

#Start API service
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]