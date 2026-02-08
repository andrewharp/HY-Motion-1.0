#!/usr/bin/env python3
import sys
import os

# Prevent segfault
os.environ['LD_PRELOAD'] = ''

sys.path.insert(0, '/workspace/hymotion_api')
sys.path.insert(0, '/workspace/ComfyUI-HyMotion')

print('Importing pipeline...')
from hymotion_api_pipeline import HYMotionRetargetingPipeline

print('Creating pipeline instance...')
pipeline = HYMotionRetargetingPipeline(
    model_path='/workspace/HY-Motion-1.0/ckpts',
    device='cuda'
)

# Change to HY-Motion dir so relative paths work
os.chdir('/workspace/HY-Motion-1.0')

print('Loading model (this takes ~60-90 seconds)...')
pipeline.load_model()

print('Processing cyber.glb...')
result = pipeline.process(
    avatar_input='/workspace/hymotion_api/cyber.glb',
    prompt='crouches down and looks around sneakily',
    num_frames=120
)

print(f'Result: {result}')

# Copy output to web-accessible location
if result.get('success') and result.get('output_glb'):
    import shutil
    output_path = result['output_glb']
    web_path = '/workspace/output/cyber_sneaky.glb'
    shutil.copy(output_path, web_path)
    print(f'Copied to: {web_path}')
