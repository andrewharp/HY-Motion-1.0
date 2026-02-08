#!/usr/bin/env python3
"""Test the HY-Motion API pipeline without starting the full server."""
import sys
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/HY-Motion-1.0')
sys.path.insert(0, '/workspace/ComfyUI-HyMotion')

print("=" * 60)
print("HY-Motion API Pipeline Test")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from hymotion_api_pipeline import HYMotionRetargetingPipeline
    print("   ✓ hymotion_api_pipeline imported")
except Exception as e:
    print(f"   ✗ Failed to import hymotion_api_pipeline: {e}")
    sys.exit(1)

try:
    from validation import validate_skeleton_integrity
    print("   ✓ validation imported")
except Exception as e:
    print(f"   ✗ Failed to import validation: {e}")

try:
    from hymotion.utils.t2m_runtime import T2MRuntime
    print("   ✓ T2MRuntime imported")
except Exception as e:
    print(f"   ✗ Failed to import T2MRuntime: {e}")

try:
    from hymotion.utils.retarget_fbx import load_fbx, get_skeleton_height
    print("   ✓ FBX utils imported")
except Exception as e:
    print(f"   ✗ Failed to import FBX utils: {e}")

# Test CUDA
print("\n2. Testing CUDA...")
import torch
if torch.cuda.is_available():
    print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("   ✗ CUDA not available (will use CPU)")

# Test model loading (this takes time, so skip for quick test)
print("\n3. Pipeline instantiation...")
try:
    pipeline = HYMotionRetargetingPipeline()
    print("   ✓ Pipeline created")
    print(f"   - Model path: {pipeline.model_path}")
    print(f"   - Device: {pipeline.device}")
    print(f"   - Temp dir: {pipeline.temp_dir}")
except Exception as e:
    print(f"   ✗ Failed to create pipeline: {e}")

print("\n" + "=" * 60)
print("Basic test complete!")
print("To test model loading (takes ~60s), run:")
print("  python3 -c \"from hymotion_api_pipeline import HYMotionRetargetingPipeline; p = HYMotionRetargetingPipeline(); p.load_model()\"")
print("=" * 60)
