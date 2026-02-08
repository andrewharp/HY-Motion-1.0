# HY-Motion-1.0 Dockerfile
# Target platform: x86_64 with CUDA 12.8 and PyTorch 2.8.0

# Using RunPod PyTorch base image with CUDA 12.8.1 and PyTorch 2.8.0
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

LABEL maintainer="HY-Motion-1.0 Docker Build"
LABEL description="HY-Motion-1.0 Text-to-3D Motion Generation"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Set up environment variables for model paths
ENV USE_HF_MODELS=0
ENV DISABLE_PROMPT_ENGINEERING=True
ENV CKPTS_DIR=/app/ckpts
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libgtk-3-0 \
    libgl1 \
    build-essential \
    protobuf-compiler \
    libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Set working directory
WORKDIR /app

# Install Python dependencies
# Note: PyTorch is already installed in the base image
RUN pip install --upgrade pip setuptools wheel

# Install core ML dependencies (without torch/torchvision as they're in base image)
RUN pip install --no-cache-dir \
    huggingface_hub==0.30.0 \
    torchdiffeq==0.2.5 \
    accelerate==0.30.1 \
    diffusers==0.26.3 \
    transformers==4.53.3 \
    einops==0.8.1 \
    safetensors==0.5.3

# Install additional dependencies
RUN pip install --no-cache-dir \
    "numpy>=1.24.0,<2.0" \
    scipy>=1.10.0 \
    transforms3d==0.4.2 \
    PyYAML==6.0 \
    omegaconf==2.3.0 \
    click==8.1.3 \
    requests==2.32.4 \
    openai==1.78.1 \
    protobuf

# Install Gradio and other UI dependencies
# Note: gradio may upgrade huggingface_hub, so we pin it after
RUN pip install --no-cache-dir \
    gradio>=4.0.0 \
    spaces \
    packaging && \
    pip install --no-cache-dir --force-reinstall \
    huggingface_hub==0.30.0

# Clone the HY-Motion-1.0 repository
RUN git clone https://github.com/Tencent-Hunyuan/HY-Motion-1.0.git /app/HY-Motion-1.0

# Set working directory to the repo
WORKDIR /app/HY-Motion-1.0

# Create model checkpoints directory structure
RUN mkdir -p /app/ckpts/tencent/HY-Motion-1.0 \
    /app/ckpts/tencent/HY-Motion-1.0-Lite \
    /app/ckpts/clip-vit-large-patch14 \
    /app/ckpts/Qwen3-8B \
    /app/.cache/huggingface

# Create startup script for downloading models and running inference
RUN cat > /app/startup.sh << 'EOF'
#!/bin/bash
set -e

echo "============================================"
echo "HY-Motion-1.0 Docker Container"
echo "============================================"

# Check if models exist, if not download them
if [ ! -f "/app/ckpts/tencent/HY-Motion-1.0/config.yml" ]; then
    echo ">>> Downloading HY-Motion-1.0 model..."
    huggingface-cli download tencent/HY-Motion-1.0 --include "HY-Motion-1.0/*" --local-dir /app/ckpts/tencent || true
fi

if [ ! -d "/app/ckpts/tencent/HY-Motion-1.0-Lite" ]; then
    echo ">>> Downloading HY-Motion-1.0-Lite model..."
    huggingface-cli download tencent/HY-Motion-1.0 --include "HY-Motion-1.0-Lite/*" --local-dir /app/ckpts/tencent || true
fi

echo ">>> Starting HY-Motion-1.0..."
echo ">>> Available commands:"
echo "  - python3 gradio_app.py          # Launch web UI"
echo "  - python3 local_infer.py --help  # CLI inference"
echo ""

# Run the provided command or default to bash
if [ $# -eq 0 ]; then
    echo ">>> Starting interactive shell..."
    exec /bin/bash
else
    echo ">>> Running: $@"
    exec "$@"
fi
EOF

RUN chmod +x /app/startup.sh

# Set permissions
RUN chmod -R 777 /app/.cache

# Expose ports
# 7860 - Gradio web interface
# 8000 - Optional API server
EXPOSE 7860 8000

# Set environment path
ENV PATH="/app/HY-Motion-1.0:${PATH}"

# Default command
ENTRYPOINT ["/app/startup.sh"]
CMD ["/bin/bash"]
