#!/usr/bin/env python3
import sys
import os
import tempfile
import shutil
import numpy as np

os.environ['LD_PRELOAD'] = ''
os.chdir('/workspace/HY-Motion-1.0')

sys.path.insert(0, '/workspace/hymotion_api')
sys.path.insert(0, '/workspace/ComfyUI-HyMotion')

print('Importing pipeline...')
from hymotion_api_pipeline import HYMotionRetargetingPipeline
from hymotion.utils.smplh2woodfbx import SMPLH2WoodFBX
from hymotion.utils.retarget_fbx import (
    load_fbx, get_skeleton_height, extract_animation, 
    retarget_animation, apply_retargeted_animation, save_fbx,
    load_bone_mapping, collect_skeleton_nodes, Skeleton
)
from fbx import FbxTime

print('Creating pipeline...')
pipeline = HYMotionRetargetingPipeline()

# Load motion from NPZ
npz_path = '/workspace/HY-Motion-1.0/output/test_bounce/bounce_test_000.npz'
print(f'Loading motion from {npz_path}...')
data = np.load(npz_path)
motion_data = {
    'poses': data['poses'],
    'trans': data['trans'],
    'keypoints3d': data['keypoints3d']
}

# Convert GLB to T-pose FBX
cyb_path = '/workspace/hymotion_api/cyber.glb'
print(f'Converting {cyb_path} to T-pose FBX...')
tpose_fbx = pipeline.glb_to_tpose_fbx(cyb_path)

# Retarget
print('Retargeting motion...')
animated_fbx, info = pipeline.retarget_motion_to_fbx(motion_data, tpose_fbx)

# Validate
print('Validating...')
validation = pipeline.validate_skeleton(tpose_fbx, animated_fbx)
print(f'Validation: {validation}')

# Convert to GLB
print('Converting to GLB...')
output_glb = pipeline.fbx_to_glb(animated_fbx, preserve_textures=True)

print(f'Output: {output_glb}')

# Copy to web
web_path = '/workspace/output/cyber_bounce.glb'
shutil.copy(output_glb, web_path)
print(f'Copied to: {web_path}')
print('Done!')
