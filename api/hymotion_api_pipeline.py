#!/usr/bin/env python3
"""
HY-Motion-1.0 → Mixamo Retargeting API Pipeline
Complete pipeline from GLB + Text Prompt → Animated GLB
"""
import sys
import os
import tempfile
import shutil
import base64
import requests
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import subprocess
import json

# Add paths for HY-Motion and ComfyUI-HyMotion (Docker uses /app/)
sys.path.insert(0, '/app/HY-Motion-1.0')
sys.path.insert(0, '/app/ComfyUI-HyMotion')
sys.path.insert(0, '/workspace/HY-Motion-1.0')  # Fallback for volume mounts
sys.path.insert(0, '/workspace/ComfyUI-HyMotion')

from hymotion.utils.t2m_runtime import T2MRuntime
from hymotion.utils.smplh2woodfbx import SMPLH2WoodFBX
from hymotion.utils.retarget_fbx import (
    load_fbx, get_skeleton_height, extract_animation, 
    retarget_animation, apply_retargeted_animation, save_fbx,
    load_bone_mapping, collect_skeleton_nodes, Skeleton, BoneData
)
from fbx import FbxTime


class HYMotionRetargetingPipeline:
    """Complete pipeline: GLB + Text → Animated GLB with validation."""
    
    def __init__(self, model_path=None, device='cuda'):
        # Prioritize network drive (/workspace) over Docker image (/app)
        if model_path is None:
            # Check network drive first
            if os.path.exists('/workspace/HY-Motion-1.0/ckpts'):
                model_path = '/workspace/HY-Motion-1.0/ckpts'
            else:
                model_path = '/app/HY-Motion-1.0/ckpts'
        # Verify the path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        self.model_path = model_path
        self.device = device
        self.runtime = None
        
        # Check wooden template location
        if os.path.exists('/app/ComfyUI-HyMotion/assets/wooden_models/boy_Rigging_smplx_tex.fbx'):
            self.wooden_template = '/app/ComfyUI-HyMotion/assets/wooden_models/boy_Rigging_smplx_tex.fbx'
        else:
            self.wooden_template = '/workspace/ComfyUI-HyMotion/assets/wooden_models/boy_Rigging_smplx_tex.fbx'
        
        self.blender_path = '/usr/bin/blender'
        self.temp_dir = tempfile.mkdtemp(prefix='hymotion_api_')
        
    def load_model(self, model_name='HY-Motion-1.0'):
        """Load the motion generation model."""
        if self.runtime is None:
            print(f"Loading {model_name}...")
            # Find config path
            config_path = os.path.join(self.model_path, model_name, 'config.yml')
            if not os.path.exists(config_path):
                # Try alternative locations
                alt_paths = [
                    '/workspace/HY-Motion-1.0/ckpts',
                    '/app/HY-Motion-1.0/ckpts',
                ]
                for alt_path in alt_paths:
                    test_path = os.path.join(alt_path, model_name, 'config.yml')
                    if os.path.exists(test_path):
                        config_path = test_path
                        break
            
            print(f"Config path: {config_path}")
            
            # Load model on GPU with text encoder on CPU to save memory
            self.runtime = T2MRuntime(
                config_path=config_path,
                skip_text=False,
                device_ids=[0],  # Use GPU for motion model
                disable_prompt_engineering=True
            )
            
            # Keep text encoder on CPU to save GPU memory
            if hasattr(self.runtime, 'pipelines') and self.runtime.pipelines:
                for pipeline in self.runtime.pipelines:
                    if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
                        print("Keeping text encoder on CPU to save GPU memory...")
                        pipeline.text_encoder = pipeline.text_encoder.to('cpu')
                        # Add a flag to indicate text encoder is on CPU
                        pipeline._text_encoder_on_cpu = True
                
            print("Model loaded!")
        return self
    
    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _get_temp_path(self, suffix: str) -> str:
        """Generate a temporary file path."""
        return os.path.join(self.temp_dir, f"{os.urandom(8).hex()}{suffix}")
    
    def download_glb(self, url: str) -> str:
        """Download GLB from URL to temp file."""
        local_path = self._get_temp_path('.glb')
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(response.content)
        return local_path
    
    def decode_base64_glb(self, b64_data: str) -> str:
        """Decode base64 GLB to temp file."""
        local_path = self._get_temp_path('.glb')
        # Handle data URL prefix
        if ',' in b64_data:
            b64_data = b64_data.split(',')[1]
        with open(local_path, 'wb') as f:
            f.write(base64.b64decode(b64_data))
        return local_path
    
    def glb_to_tpose_fbx(self, glb_path: str) -> str:
        """Convert GLB to T-pose FBX using Blender."""
        fbx_path = self._get_temp_path('_tpose.fbx')
        
        script_content = f'''
import bpy
import sys

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import GLB
bpy.ops.import_scene.gltf(filepath="{glb_path}")

# Find armature
armature = None
for obj in bpy.context.scene.objects:
    if obj.type == 'ARMATURE':
        armature = obj
        break

if armature:
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    # Apply T-pose rotations (swing arms up from A-pose)
    import math
    for side in ['L', 'R']:
        bone_name = f'{{side}}eftArm'
        if bone_name in armature.pose.bones:
            pbone = armature.pose.bones[bone_name]
            angle = math.radians(-45) if side == 'L' else math.radians(45)
            pbone.rotation_euler = (0, 0, angle)
            bpy.context.view_layer.update()
    
    # Apply pose as rest pose
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

# Export FBX with textures
bpy.ops.export_scene.fbx(
    filepath="{fbx_path}",
    use_selection=False,
    embed_textures=True,
    path_mode='COPY'
)
print(f"Exported: {fbx_path}")
'''
        script_path = self._get_temp_path('.py')
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        result = subprocess.run(
            [self.blender_path, '--background', '--python', script_path],
            capture_output=True, text=True, timeout=120
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Blender GLB→FBX conversion failed: {result.stderr}")
        
        return fbx_path
    
    def fbx_to_glb(self, fbx_path: str) -> str:
        """Convert animated FBX to GLB using Blender."""
        glb_path = self._get_temp_path('_animated.glb')
        
        script_content = f'''
import bpy
import sys

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import FBX
bpy.ops.import_scene.fbx(filepath="{fbx_path}")

# Export GLB with animations
bpy.ops.export_scene.gltf(
    filepath="{glb_path}",
    export_format='GLB',
    export_animations=True,
    export_animation_mode='SCENE',
    export_yup=True,
    export_materials='EXPORT'
)
print(f"Exported: {glb_path}")
'''
        script_path = self._get_temp_path('.py')
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        result = subprocess.run(
            [self.blender_path, '--background', '--python', script_path],
            capture_output=True, text=True, timeout=120
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Blender FBX→GLB conversion failed: {result.stderr}")
        
        return glb_path
    
    def merge_textured_with_animation(self, textured_glb_path: str, animated_glb_path: str) -> str:
        """Merge textures from original GLB with animation from animated GLB."""
        output_path = self._get_temp_path('_final.glb')
        
        script_content = f'''
import bpy
import sys

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import textured GLB (for mesh and materials)
bpy.ops.import_scene.gltf(filepath="{textured_glb_path}")
textured_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

# Store materials and UVs
material_map = {{}}
for obj in textured_objects:
    if obj.data.materials:
        material_map[obj.name] = [mat for mat in obj.data.materials]

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import animated GLB (for animation)
bpy.ops.import_scene.gltf(filepath="{animated_glb_path}")
animated_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
animated_armature = None
for obj in bpy.context.scene.objects:
    if obj.type == 'ARMATURE':
        animated_armature = obj
        break

# Restore materials from textured version
for obj in animated_objects:
    base_name = obj.name.replace('.001', '').replace('.002', '')
    if base_name in material_map:
        obj.data.materials.clear()
        for mat in material_map[base_name]:
            obj.data.materials.append(mat)

# Export merged GLB
bpy.ops.export_scene.gltf(
    filepath="{output_path}",
    export_format='GLB',
    export_animations=True,
    export_animation_mode='SCENE',
    export_yup=True,
    export_materials='EXPORT'
)
print(f"Exported: {output_path}")
'''
        script_path = self._get_temp_path('.py')
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        result = subprocess.run(
            [self.blender_path, '--background', '--python', script_path],
            capture_output=True, text=True, timeout=120
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Blender merge failed: {result.stderr}")
        
        return output_path
    
    def generate_motion_from_prompt(self, prompt: str, num_frames: int = 120, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Generate motion from text prompt."""
        if self.runtime is None:
            self.load_model()
        
        print(f"Generating motion: '{prompt}'")
        
        # Calculate duration from frames (30 FPS)
        duration = num_frames / 30.0
        
        # Generate motion using T2MRuntime
        seeds = [seed if seed is not None else 42]
        seeds_csv = ",".join(map(str, seeds))
        
        html, fbx_files, motion_data = self.runtime.generate_motion(
            text=prompt,
            seeds_csv=seeds_csv,
            duration=duration,
            cfg_scale=5.0,
            output_format='dict'  # Get motion data as dict, not FBX
        )
        
        # Extract motion data
        output_dict = motion_data.output_dict
        
        # Convert to the expected format
        # rot6d: [B, L, J, 6], transl: [B, L, 3]
        rot6d = output_dict['rot6d'][0]  # Take first batch [L, J, 6]
        transl = output_dict['transl'][0]  # [L, 3]
        keypoints3d = output_dict.get('keypoints3d', [None])[0]  # [L, 52, 3] if available
        
        # Convert rot6d to axis-angle format for SMPL-H (156 dims for poses)
        num_frames_actual = rot6d.shape[0]
        
        # Construct poses array [T, 156] (SMPL-H format)
        poses = np.zeros((num_frames_actual, 156))
        
        result = {
            'poses': poses,  # Placeholder - actual conversion done in retargeting
            'trans': transl,
            'keypoints3d': keypoints3d if keypoints3d is not None else np.zeros((num_frames_actual, 52, 3)),
            '_raw_output': output_dict,  # Keep raw for wooden converter
            '_motion_data': motion_data
        }
        
        print(f"Generated {num_frames_actual} frames")
        return result
    
    def generate_motion_from_npz_url(self, npz_url: str) -> Dict[str, np.ndarray]:
        """Load motion from NPZ URL."""
        response = requests.get(npz_url, timeout=60)
        response.raise_for_status()
        
        temp_npz = self._get_temp_path('.npz')
        with open(temp_npz, 'wb') as f:
            f.write(response.content)
        
        data = np.load(temp_npz)
        result = {
            'poses': data['poses'],
            'trans': data['trans'],
            'keypoints3d': data['keypoints3d']
        }
        os.unlink(temp_npz)
        return result
    
    def retarget_motion_to_fbx(self, motion_data: Dict[str, np.ndarray], 
                                target_fbx_path: str,
                                yaw_offset: float = 0.0,
                                in_place: bool = False) -> Tuple[str, Dict]:
        """Retarget motion to target FBX skeleton."""
        # Step 1: Motion → Wooden FBX
        print("Step 1: Motion → Wooden FBX")
        
        wooden_output = self._get_temp_path('_wooden.fbx')
        
        # Check if we have raw motion data from T2MRuntime
        if '_motion_data' in motion_data and self.runtime and self.runtime.fbx_available:
            # Use runtime's FBX converter directly
            m_data = motion_data['_motion_data']
            output_dict = m_data.output_dict
            
            # Use the runtime's FBX converter
            if hasattr(self.runtime, 'fbx_converter') and self.runtime.fbx_converter:
                success = self.runtime.fbx_converter.convert_output_to_fbx(
                    output_dict,
                    wooden_output,
                    fps=30
                )
                if not success:
                    raise RuntimeError("Failed to convert motion to wooden FBX using runtime converter")
            else:
                # Fallback: save as NPZ and convert
                temp_npz = self._get_temp_path('.npz')
                np.savez(temp_npz, 
                         poses=motion_data['poses'],
                         trans=motion_data['trans'],
                         keypoints3d=motion_data['keypoints3d'])
                
                converter = SMPLH2WoodFBX(
                    template_fbx_path=self.wooden_template,
                    scale=100,
                )
                
                success = converter.convert_npz_to_fbx(
                    temp_npz,
                    wooden_output,
                    fps=30,
                    absolute_root=True
                )
                os.unlink(temp_npz)
                
                if not success:
                    raise RuntimeError("Failed to convert to wooden FBX")
        else:
            # Traditional NPZ path
            temp_npz = self._get_temp_path('.npz')
            np.savez(temp_npz, 
                     poses=motion_data['poses'],
                     trans=motion_data['trans'],
                     keypoints3d=motion_data['keypoints3d'])
            
            converter = SMPLH2WoodFBX(
                template_fbx_path=self.wooden_template,
                scale=100,
            )
            
            success = converter.convert_npz_to_fbx(
                temp_npz,
                wooden_output,
                fps=30,
                absolute_root=True
            )
            os.unlink(temp_npz)
            
            if not success:
                raise RuntimeError("Failed to convert to wooden FBX")
        
        print(f"Created intermediate: {wooden_output}")
        
        # Step 2: Wooden FBX → Target FBX (Retargeting)
        print("Step 2: Wooden FBX → Target FBX (Retargeting)")
        
        # Load source (animated wooden)
        src_man, src_scene, src_skel = load_fbx(wooden_output, sample_rest_frame=None)
        
        if get_skeleton_height(src_skel) < 0.1:
            src_man, src_scene, src_skel = load_fbx(wooden_output, sample_rest_frame=0)
            collect_skeleton_nodes(src_scene.GetRootNode(), src_skel, sampling_time=FbxTime())
        
        extract_animation(src_scene, src_skel)
        src_bone_count = len(src_skel.bones)
        src_height = get_skeleton_height(src_skel)
        print(f"Source: {src_bone_count} bones, height={src_height:.2f}")
        
        # Load target (T-pose)
        tgt_man, tgt_scene, tgt_skel = load_fbx(target_fbx_path)
        tgt_bone_count = len(tgt_skel.bones)
        tgt_height = get_skeleton_height(tgt_skel)
        print(f"Target: {tgt_bone_count} bones, height={tgt_height:.2f}")
        
        # Get bone mapping
        mapping = load_bone_mapping('', src_skel, tgt_skel)
        
        # Retarget
        rots, locs, active = retarget_animation(
            src_skel, tgt_skel, mapping,
            force_scale=0.0,
            yaw_offset=yaw_offset,
            neutral_fingers=True,
            in_place=in_place,
            auto_stride=False,
            preserve_position=False
        )
        
        print(f"Retargeted {len(active)} bones")
        
        # Apply animation
        src_time_mode = src_scene.GetGlobalSettings().GetTimeMode()
        apply_retargeted_animation(
            tgt_scene, tgt_skel, rots, locs,
            src_skel.frame_start, src_skel.frame_end,
            src_time_mode
        )
        
        # Save
        output_fbx = self._get_temp_path('_retargeted.fbx')
        save_fbx(tgt_man, tgt_scene, output_fbx)
        
        # Cleanup wooden temp
        if os.path.exists(wooden_output):
            os.remove(wooden_output)
        
        print(f"Retargeted FBX: {output_fbx}")
        return output_fbx, {
            'source_bones': src_bone_count,
            'target_bones': tgt_bone_count,
            'source_height': src_height,
            'target_height': tgt_height,
            'retargeted_bones': len(active)
        }
    
    def validate_skeleton(self, input_fbx_path: str, output_fbx_path: str) -> Dict[str, Any]:
        """Validate retargeting by comparing input and output skeletons."""
        # Load input skeleton
        _, input_scene, input_skel = load_fbx(input_fbx_path)
        input_height = get_skeleton_height(input_skel)
        input_bones = len(input_skel.bones)
        
        # Load output skeleton
        _, output_scene, output_skel = load_fbx(output_fbx_path)
        extract_animation(output_scene, output_skel)
        output_height = get_skeleton_height(output_skel)
        output_bones = len(output_skel.bones)
        
        # Validation checks
        validation = {
            'input_bone_count': input_bones,
            'output_bone_count': output_bones,
            'input_height': round(input_height, 4),
            'output_height': round(output_height, 4),
            'bone_count_match': input_bones == output_bones,
            'height_ratio': round(output_height / input_height, 4) if input_height > 0 else 0,
            'valid': True,
            'errors': []
        }
        
        # Check bone count
        if input_bones != output_bones:
            validation['valid'] = False
            validation['errors'].append(f"Bone count mismatch: {input_bones} vs {output_bones}")
        
        # Check height ratio (should be ~1.0 with small tolerance)
        if validation['height_ratio'] < 0.5 or validation['height_ratio'] > 2.0:
            validation['valid'] = False
            validation['errors'].append(f"Height ratio out of bounds: {validation['height_ratio']}")
        
        # Check for reasonable skeleton height
        if output_height < 10:  # Less than 10cm is suspicious
            validation['valid'] = False
            validation['errors'].append(f"Output skeleton height too small: {output_height}cm")
        
        return validation
    
    def process(self, 
                avatar_input: str,  # URL or base64 GLB
                prompt: Optional[str] = None,
                npz_url: Optional[str] = None,
                num_frames: int = 120,
                seed: Optional[int] = None,
                yaw_offset: float = 0.0,
                in_place: bool = False,
                preserve_textures: bool = True) -> Dict[str, Any]:
        """
        Main pipeline: GLB + Motion → Animated GLB
        
        Args:
            avatar_input: URL or base64-encoded GLB
            prompt: Text prompt for motion generation
            npz_url: URL to NPZ file (alternative to prompt)
            num_frames: Number of frames to generate
            seed: Random seed for motion generation
            yaw_offset: Rotation around Y axis
            in_place: Remove horizontal movement
            preserve_textures: Whether to preserve original textures
            
        Returns:
            Dict with result status, output GLB path, and validation info
        """
        try:
            # Step 1: Get input GLB
            print("=" * 60)
            print("Step 1: Get input GLB")
            print("=" * 60)
            
            if avatar_input.startswith('http'):
                input_glb_path = self.download_glb(avatar_input)
            elif avatar_input.startswith('data:') or len(avatar_input) > 1000:
                input_glb_path = self.decode_base64_glb(avatar_input)
            else:
                input_glb_path = avatar_input
            
            print(f"Input GLB: {input_glb_path}")
            
            # Step 2: Convert GLB to T-pose FBX
            print("=" * 60)
            print("Step 2: Convert GLB to T-pose FBX")
            print("=" * 60)
            
            tpose_fbx_path = self.glb_to_tpose_fbx(input_glb_path)
            print(f"T-pose FBX: {tpose_fbx_path}")
            
            # Step 3: Get motion data
            print("=" * 60)
            print("Step 3: Get motion data")
            print("=" * 60)
            
            if npz_url:
                motion_data = self.generate_motion_from_npz_url(npz_url)
            elif prompt:
                motion_data = self.generate_motion_from_prompt(prompt, num_frames, seed)
            else:
                raise ValueError("Either prompt or npz_url must be provided")
            
            # Step 4: Retarget motion
            print("=" * 60)
            print("Step 4: Retarget motion")
            print("=" * 60)
            
            animated_fbx_path, retarget_info = self.retarget_motion_to_fbx(
                motion_data, tpose_fbx_path, yaw_offset, in_place
            )
            
            # Step 5: Validate
            print("=" * 60)
            print("Step 5: Validate retargeting")
            print("=" * 60)
            
            validation = self.validate_skeleton(tpose_fbx_path, animated_fbx_path)
            validation.update(retarget_info)
            
            if not validation['valid']:
                return {
                    'success': False,
                    'error': 'Validation failed',
                    'validation': validation
                }
            
            # Step 6: Convert to GLB
            print("=" * 60)
            print("Step 6: Convert to GLB")
            print("=" * 60)
            
            if preserve_textures:
                # Convert animated FBX to GLB first
                animated_glb_path = self.fbx_to_glb(animated_fbx_path)
                # Merge textures from original with animation
                final_glb_path = self.merge_textured_with_animation(input_glb_path, animated_glb_path)
            else:
                final_glb_path = self.fbx_to_glb(animated_fbx_path)
            
            print(f"Final GLB: {final_glb_path}")
            
            return {
                'success': True,
                'output_glb_path': final_glb_path,
                'validation': validation,
                'temp_dir': self.temp_dir
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'validation': {}
            }
