FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# All service modules
COPY *.py /app/

# Pipeline modules (chunk_types, chunk_extractors)
COPY pipeline/ /app/pipeline/

# OCI config + Oracle wallet
COPY Configurations/ /app/Configurations/
COPY oci_config /app/oci_config
COPY wallet/ /app/wallet/

ENV PIPELINE_DIR=/app/pipeline
ENV PDF_ROOT=/app/data
ENV MD_OUT_DIR=/app/md_output
ENV OCI_KEY_FILE=/app/Configurations/oci/oci_api_key.pem
ENV OCI_CONFIG_FILE=/app/oci_config
ENV ORACLE_WALLET_DIR=/app/wallet
ENV ORACLE_WALLET_PASSWORD=""

# Default: ingestion service. Override in docker-compose for RAG.
ENV SERVICE_PORT=8074
ENV RAG_PORT=7874

EXPOSE 8074 7874

CMD ["python", "service.py"]
