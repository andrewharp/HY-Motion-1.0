#!/bin/bash
# Setup HY-Motion API pod with network volume for models

# Create pod with 20GB network volume
runpodctl create pod \
  --name "hymotion-api-v2" \
  --gpuType "NVIDIA RTX A4000" \
  --imageName "ghcr.io/andrewharp/hymotion-api:latest" \
  --volumeSize 20 \
  --volumePath "/workspace" \
  --env "CKPTS_DIR=/workspace/HY-Motion-1.0/ckpts"

echo "Pod created. Once running, SSH in and run:"
echo "  python3 /app/HY-Motion-1.0/api/download_models.py"
