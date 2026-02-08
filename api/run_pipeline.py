#!/usr/bin/env python3
import sys
import os

# Prevent segfault by setting environment
os.environ['LD_PRELOAD'] = ''

sys.path.insert(0, '/workspace/hymotion_api')
sys.path.insert(0, '/workspace/ComfyUI-HyMotion')

print('Importing pipeline...')
from hymotion_api_pipeline import HYMotionRetargetingPipeline

print('Creating pipeline instance...')
pipeline = HYMotionRetargetingPipeline()

print('Loading model (this takes ~60-90 seconds)...')
pipeline.load_model()

print('Processing cyber.glb...')
result = pipeline.process(
    avatar_input='/workspace/hymotion_api/cyber.glb',
    prompt='crouches down and looks around sneakily',
    num_frames=120
)

print(f'Result: {result}')
