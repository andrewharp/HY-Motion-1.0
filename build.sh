#!/bin/bash
# Build script for HY-Motion-1.0 Docker image on Jetson ARM64

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "HY-Motion-1.0 Docker Build Script"
echo "Target Platform: linux/arm64 (Jetson)"
echo "============================================"

# Detect architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

if [ "$ARCH" != "aarch64" ]; then
    echo "Warning: This Dockerfile is optimized for ARM64/aarch64 (Jetson)"
    echo "Current architecture: $ARCH"
    echo "Build may still work but could have compatibility issues."
fi

# Set image name and tag
IMAGE_NAME="hymotion"
IMAGE_TAG="1.0-jetson"
FULL_TAG="${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo "Building Docker image: ${FULL_TAG}"
echo "This may take 15-30 minutes depending on network speed..."
echo ""

# Build the Docker image
# --platform linux/arm64 ensures we build for ARM64
docker build \
    --platform linux/arm64 \
    --tag "${FULL_TAG}" \
    --tag "${IMAGE_NAME}:latest" \
    --progress=plain \
    . 2>&1 | tee build.log

BUILD_STATUS=${PIPESTATUS[0]}

if [ $BUILD_STATUS -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✓ Build completed successfully!"
    echo "============================================"
    echo ""
    echo "Image details:"
    docker images "${FULL_TAG}" --format "  Name: {{.Repository}}:{{.Tag}}\\n  Size: {{.Size}}\\n  Created: {{.CreatedAt}}"
    echo ""
    echo "Next steps:"
    echo "  1. Run with Gradio UI:  docker compose up -d"
    echo "  2. Run CLI inference:   docker run -it --rm --gpus all ${FULL_TAG} python3 local_infer.py --help"
    echo "  3. Interactive shell:   docker run -it --rm --gpus all ${FULL_TAG}"
    echo ""
else
    echo ""
    echo "============================================"
    echo "✗ Build failed with exit code: ${BUILD_STATUS}"
    echo "============================================"
    echo "Check build.log for details"
    exit $BUILD_STATUS
fi
