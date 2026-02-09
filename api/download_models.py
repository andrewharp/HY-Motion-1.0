#!/usr/bin/env python3
"""
Download HY-Motion-1.0 models to network volume.
Run this on the pod to download models if they don't exist.
"""

import os
import sys
from pathlib import Path

def download_models():
    """Download models from HuggingFace to /workspace/HY-Motion-1.0/ckpts/"""
    
    # Base path for models (network volume)
    base_path = Path("/workspace/HY-Motion-1.0/ckpts")
    base_path.mkdir(parents=True, exist_ok=True)
    
    print("=== Downloading HY-Motion-1.0 models ===")
    print(f"Target directory: {base_path}")
    
    # Check if models already exist
    tencent_path = base_path / "tencent" / "HY-Motion-1.0"
    if (tencent_path / "config.yml").exists():
        print("✓ Models already exist, skipping download")
        return 0
    
    try:
        from huggingface_hub import snapshot_download
        
        # Download HY-Motion-1.0 models
        print("Downloading tencent/HY-Motion-1.0...")
        snapshot_download(
            repo_id="tencent/HY-Motion-1.0",
            local_dir=str(base_path / "tencent" / "HY-Motion-1.0"),
            local_dir_use_symlinks=False
        )
        print("✓ HY-Motion-1.0 models downloaded")
        
        # Download CLIP
        print("Downloading openai/clip-vit-large-patch14...")
        snapshot_download(
            repo_id="openai/clip-vit-large-patch14",
            local_dir=str(base_path / "clip-vit-large-patch14"),
            local_dir_use_symlinks=False
        )
        print("✓ CLIP downloaded")
        
        print("\n=== All models downloaded successfully ===")
        return 0
        
    except Exception as e:
        print(f"✗ Error downloading models: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(download_models())
