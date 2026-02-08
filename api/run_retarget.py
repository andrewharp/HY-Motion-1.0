#!/usr/bin/env python3
import sys
import os
import tempfile
import shutil

os.environ['LD_PRELOAD'] = ''
os.chdir('/workspace/HY-Motion-1.0')

sys.path.insert(0, '/workspace/hymotion_api')
sys.path.insert(0, '/workspace/ComfyUI-HyMotion')

print('Importing pipeline...')
from hymotion_api_pipeline import HYMotionRetargetingPipeline

print('Creating pipeline...')
pipeline = HYMotionRetargetingPipeline()

# Don't load T2MRuntime - use NPZ directly
print('Processing cyber.glb with existing NPZ...')

# Use existing bounce NPZ
npz_path = '/workspace/HY-Motion-1.0/output/test_bounce/bounce_test_000.npz'

result = pipeline.process(
    avatar_input='/workspace/hymotion_api/cyber.glb',
    npz_url=None,  # We'll load manually
    prompt=None,
    num_frames=120
)

# Actually, let's just call the retarget method directly
print('Loading motion from NPZ...')
import numpy as np
motion_data = pipeline.generate_motion_from_npz_file(npz_path)

print('Converting GLB to T-pose FBX...')
tpose_fbx = pipeline.glb_to_tpose_fbx('/workspace/hymotion_api/cyber.glb')

print('Retargeting...')
animated_fbx, info = pipeline.retarget_motion_to_fbx(motion_data, tpose_fbx)

print('Converting to GLB...')
output_glb = pipeline.fbx_to_glb(animated_fbx, preserve_textures=True)

print(f'Output: {output_glb}')

# Copy to web
web_path = '/workspace/output/cyber_bounce.glb'
shutil.copy(output_glb, web_path)
print(f'Copied to: {web_path}')
