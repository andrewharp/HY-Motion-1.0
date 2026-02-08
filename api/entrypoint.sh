#!/bin/bash
# Entrypoint script for HY-Motion API on RunPod
# Calls base image startup scripts and then starts our API

set -e

echo "=== HY-Motion API Entrypoint ==="

# RunPod base image uses PUBLIC_KEY env var to setup SSH
# If PUBLIC_KEY is set, SSH will be started automatically

# Source any RunPod initialization scripts
if [ -f /etc/runpod/start.sh ]; then
    echo "Sourcing RunPod start script..."
    source /etc/runpod/start.sh
elif [ -f /start.sh ]; then
    echo "Sourcing /start.sh..."
    source /start.sh
fi

# If PUBLIC_KEY is provided but SSH isn't running, start it manually
if [ -n "$PUBLIC_KEY" ] && ! pgrep -x "sshd" > /dev/null; then
    echo "Setting up SSH with provided PUBLIC_KEY..."
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys
    service ssh start || /usr/sbin/sshd -D &
    echo "SSH daemon started on port 22"
fi

# Wait a moment for services to initialize
sleep 2

# Check what's running
echo "Running processes:"
ps aux | grep -E "(ssh|python)" | grep -v grep || true

# Start the API service
echo "Starting HY-Motion API service..."
exec python3 /app/api/api_service.py
