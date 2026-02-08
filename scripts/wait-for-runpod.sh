#!/bin/bash
# RunPod Deployment Watcher - handles manifest propagation delays
set -e

IMAGE_URL="$1"
ENDPOINT_NAME="${2:-hymotion-serverless}"
TIMEOUT="${3:-1800}"

[ -z "$IMAGE_URL" ] && { echo "Usage: $0 <image_url> [endpoint_name] [timeout]"; exit 1; }

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_pod() { echo -e "${CYAN}[POD]${NC} $1"; }

if ! command -v runpodctl &> /dev/null; then log_error "runpodctl required"; exit 1; fi

IMAGE_TAG=$(echo "$IMAGE_URL" | rev | cut -d: -f1 | rev)
log_info "Deploying: $IMAGE_URL"
log_info "Endpoint: $ENDPOINT_NAME"

# Create or find endpoint
log_info "Checking for existing endpoint..."
EXISTING=$(runpodctl get endpoint 2>/dev/null | grep "$ENDPOINT_NAME" | awk '{print $1}' || echo "")

if [ -n "$EXISTING" ]; then
    log_info "Updating existing endpoint: $EXISTING"
    # Note: runpodctl doesn't have direct update, would need API call or manual update
    log_warn "Please update endpoint $EXISTING manually via console with image:"
    log_warn "$IMAGE_URL"
    ENDPOINT_ID=$EXISTING
else
    log_info "Creating new endpoint..."
    # Create endpoint - models baked in, so no network volume needed
    ENDPOINT_ID=$(runpodctl create endpoint \
        --name "$ENDPOINT_NAME" \
        --imageName "$IMAGE_URL" \
        --gpuType "NVIDIA GeForce RTX 4090" \
        --minWorkers 0 \
        --maxWorkers 2 \
        --containerDiskSize 25 \
        --env "MODEL_PATH=/app/ckpts" \
        --env "OUTPUT_MODE=base64" 2>&1 | grep -oE '[a-z0-9]{12,}' | head -1 || echo "")
    
    [ -z "$ENDPOINT_ID" ] && { log_error "Failed to create endpoint"; exit 1; }
    log_info "Created: $ENDPOINT_ID"
fi

# Wait for workers with manifest error handling
log_info "Waiting for workers..."
START=$(date +%s)
INTERVAL=10
MANIFEST_ERRORS=0
MAX_MANIFEST_ERRORS=20

while true; do
    ELAPSED=$(($(date +%s) - START))
    [ $ELAPSED -ge $TIMEOUT ] && { log_error "Timeout after ${TIMEOUT}s"; exit 1; }
    
    INFO=$(runpodctl get endpoint "$ENDPOINT_ID" 2>&1 || echo "")
    
    # Handle manifest unknown (GHCR propagation delay)
    if echo "$INFO" | grep -qi "manifest unknown"; then
        MANIFEST_ERRORS=$((MANIFEST_ERRORS + 1))
        printf "\r${YELLOW}[GHCR]${NC} Waiting for image propagation... %dm %ds (errors: %d/%d)" \
            $((ELAPSED/60)) $((ELAPSED%60)) $MANIFEST_ERRORS $MAX_MANIFEST_ERRORS
        [ $MANIFEST_ERRORS -ge $MAX_MANIFEST_ERRORS ] && { echo ""; log_error "Manifest timeout"; exit 1; }
        sleep $INTERVAL
        continue
    elif [ $MANIFEST_ERRORS -gt 0 ]; then
        echo ""; log_info "Image propagated successfully!"
        MANIFEST_ERRORS=0
    fi
    
    # Check for errors
    if echo "$INFO" | grep -qi "error creating container"; then
        log_error "Container error: $INFO"
        exit 1
    fi
    
    # Check if workers are running
    if echo "$INFO" | grep -qi "running\|active" && ! echo "$INFO" | grep -q "0 running"; then
        echo ""
        log_info "✓ Workers are running!"
        log_info "Endpoint: https://api.runpod.ai/v2/$ENDPOINT_ID"
        exit 0
    fi
    
    printf "\r${BLUE}[INIT]${NC} Workers starting... %02d:%02d" $((ELAPSED/60)) $((ELAPSED%60))
    INTERVAL=$((INTERVAL < 60 ? INTERVAL + 5 : 60))
    sleep $INTERVAL
done
