FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pipeline modules from azadea
COPY pipeline/ /app/pipeline/

# Service
COPY service.py /app/service.py

ENV PIPELINE_DIR=/app/pipeline
ENV PDF_ROOT=/app/data
ENV MD_OUT_DIR=/app/md_output
ENV MULTIMODAL_IMAGES_DIR=/app/images
ENV SERVICE_PORT=8073

EXPOSE 8073

CMD ["python", "service.py"]
