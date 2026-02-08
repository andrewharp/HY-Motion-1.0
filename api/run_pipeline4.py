#!/usr/bin/env python3
import sys
import os

os.environ['LD_PRELOAD'] = ''

# Change to HY-Motion dir so relative paths work
os.chdir('/workspace/HY-Motion-1.0')

sys.path.insert(0, '/workspace/hymotion_api')
sys.path.insert(0, '/workspace/ComfyUI-HyMotion')

print('Importing...')
from hymotion_api_pipeline import HYMotionRetargetingPipeline

print('Creating pipeline...')
pipeline = HYMotionRetargetingPipeline()

# Load with proper checkpoint path  
print('Loading model...')
config_path = '/workspace/HY-Motion-1.0/ckpts/HY-Motion-1.0/config.yml'
ckpt_path = '/workspace/HY-Motion-1.0/ckpts/HY-Motion-1.0/latest.ckpt'

from hymotion.utils.t2m_runtime import T2MRuntime
pipeline.runtime = T2MRuntime(
    config_path=config_path,
    ckpt_name=ckpt_path,
    skip_text=False,
    device_ids=[0],
    disable_prompt_engineering=True
)

print('Processing cyber.glb...')
result = pipeline.process(
    avatar_input='/workspace/hymotion_api/cyber.glb',
    prompt='crouches down and looks around sneakily',
    num_frames=120
)

print(f'Success: {result.get("success")}')
if result.get('success'):
    print(f'Output: {result.get("output_glb")}')
    import shutil
    src = result['output_glb']
    dst = '/workspace/output/cyber_sneaky.glb'
    shutil.copy(src, dst)
    print(f'Copied to web: {dst}')
else:
    print(f'Error: {result.get("error")}')
