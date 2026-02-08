#!/bin/bash
# Create RunPod with proper SSH and API configuration

IMAGE="ghcr.io/andrewharp/hymotion-api:b4f6171"

echo "Creating RunPod with SSH and API..."
echo "Image: $IMAGE"

# Read public key
if [ -f ~/.ssh/runpod_key.pub ]; then
    PUBKEY=$(cat ~/.ssh/runpod_key.pub)
elif [ -f ~/.runpod/ssh/RunPod-Key.pub ]; then
    PUBKEY=$(cat ~/.runpod/ssh/RunPod-Key.pub)
else
    echo "No public key found!"
    exit 1
fi

echo "Using public key: ${PUBKEY:0:50}..."

# Create pod with SSH port exposed and PUBLIC_KEY set
runpodctl create pod \
    --name "hymotion-api" \
    --gpuType "NVIDIA A100 80GB PCIe" \
    --imageName "$IMAGE" \
    --volumeSize 100 \
    --volumePath /workspace \
    --ports "8000/http" \
    --ports "22/tcp" \
    --startSSH \
    --env "PUBLIC_KEY=$PUBKEY" \
    --env "PYTHONUNBUFFERED=1" \
    --env "CKPTS_DIR=/app/HY-Motion-1.0/ckpts"

echo ""
echo "Pod created! Wait for it to start, then:"
echo "  API: https://<pod-id>-8000.proxy.runpod.net/health"
echo "  SSH: ssh -p <port> root@<ip> -i ~/.ssh/runpod_key"
