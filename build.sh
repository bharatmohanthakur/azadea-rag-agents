#!/bin/bash
# ============================================================================
# Build script for three-tier ingestion Docker images
#
# Copies pipeline modules from multimodal-rag/azadea/ into each tier's
# build context (pipeline/ folder), then runs docker compose build.
#
# Usage:
#   ./build.sh          # Copy pipeline + build all images
#   ./build.sh basic    # Build only basic tier
#   ./build.sh --copy   # Only copy pipeline, don't build
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AZADEA_DIR="/home/admincsp/multimodal-rag/azadea"

TIERS=("ingestion-basic" "ingestion-standard" "ingestion-premium" "ingestion-oci")

echo "=== Document Ingestion Docker Build ==="
echo "Pipeline source: $AZADEA_DIR"
echo ""

# --- Copy pipeline modules into each tier ---
for tier in "${TIERS[@]}"; do
    target="$SCRIPT_DIR/$tier/pipeline"
    echo "[$tier] Copying pipeline modules → $target/"

    rm -rf "$target"
    mkdir -p "$target"

    # Copy all Python files from azadea
    cp "$AZADEA_DIR"/*.py "$target/" 2>/dev/null || true

    # Copy fastembed cache if exists (BM25 model)
    if [ -d "$AZADEA_DIR/.fastembed_cache" ]; then
        cp -r "$AZADEA_DIR/.fastembed_cache" "$target/"
    fi

    echo "[$tier] Done — $(ls "$target"/*.py 2>/dev/null | wc -l) Python files copied"
done

echo ""

# --- Build Docker images ---
if [ "$1" = "--copy" ]; then
    echo "Pipeline copy complete. Skipping Docker build (--copy flag)."
    exit 0
fi

cd "$SCRIPT_DIR"

if [ -n "$1" ] && [ "$1" != "--copy" ]; then
    # Build specific tier
    service="ingestion-$1"
    echo "Building $service..."
    docker compose build "$service"
else
    # Build all
    echo "Building all tiers..."
    docker compose build
fi

echo ""
echo "=== Build complete ==="
echo "Start services: docker compose up -d"
echo "  Basic:    http://localhost:8071/docs"
echo "  Standard: http://localhost:8072/docs"
echo "  Premium:  http://localhost:8073/docs"
echo "  OCI:      http://localhost:8074/docs"
