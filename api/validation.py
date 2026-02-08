"""
Skeleton validation utilities for retargeting quality assurance.
"""
import numpy as np
from typing import Dict, List, Tuple, Any
import sys

sys.path.insert(0, '/workspace/ComfyUI-HyMotion')
from hymotion.utils.retarget_fbx import load_fbx, get_skeleton_height, Skeleton, BoneData


def get_bone_positions(skeleton: Skeleton, frame: int = 0) -> Dict[str, np.ndarray]:
    """Extract bone positions at a specific frame."""
    positions = {}
    
    for bone_name, bone in skeleton.bones.items():
        if frame in bone.world_location_animation:
            positions[bone_name] = bone.world_location_animation[frame]
        else:
            # Use rest pose position
            head = bone.head.copy()
            # Apply world matrix if available
            if bone.name in skeleton.node_world_matrices:
                world_mat = skeleton.node_world_matrices[bone.name]
                head = world_mat[:3, 3]
            positions[bone_name] = head
    
    return positions


def compare_bone_positions(input_positions: Dict[str, np.ndarray],
                           output_positions: Dict[str, np.ndarray],
                           tolerance: float = 1.0) -> Dict[str, Any]:
    """Compare bone positions between input and output skeletons."""
    errors = []
    total_error = 0.0
    compared_bones = 0
    
    for bone_name in input_positions:
        if bone_name in output_positions:
            input_pos = input_positions[bone_name]
            output_pos = output_positions[bone_name]
            distance = np.linalg.norm(input_pos - output_pos)
            total_error += distance
            compared_bones += 1
            
            if distance > tolerance:
                errors.append({
                    'bone': bone_name,
                    'distance': round(float(distance), 4),
                    'input_pos': input_pos.tolist(),
                    'output_pos': output_pos.tolist()
                })
    
    avg_error = total_error / compared_bones if compared_bones > 0 else 0
    
    return {
        'compared_bones': compared_bones,
        'total_error': round(float(total_error), 4),
        'average_error': round(float(avg_error), 4),
        'out_of_tolerance': len(errors),
        'errors': errors[:10],  # Limit to first 10 errors
        'passed': len(errors) == 0
    }


def validate_limb_lengths(skeleton: Skeleton) -> Dict[str, Any]:
    """Validate that limb lengths are reasonable and consistent."""
    issues = []
    limb_ratios = {}
    
    # Common limb pairs to check
    limb_pairs = [
        ('leftarm', 'rightarm'),
        ('leftleg', 'rightleg'),
        ('leftforearm', 'rightforearm'),
        ('leftshin', 'rightshin'),
    ]
    
    for left, right in limb_pairs:
        left_bone = skeleton.get_bone_case_insensitive(left)
        right_bone = skeleton.get_bone_case_insensitive(right)
        
        if left_bone and right_bone:
            # Calculate bone length from head to implied tail
            left_len = np.linalg.norm(left_bone.head)
            right_len = np.linalg.norm(right_bone.head)
            
            if left_len > 0 and right_len > 0:
                ratio = left_len / right_len
                limb_ratios[f"{left}/{right}"] = round(float(ratio), 4)
                
                # Check if ratio is within reasonable bounds (0.8 - 1.2)
                if ratio < 0.8 or ratio > 1.2:
                    issues.append({
                        'limbs': f"{left} vs {right}",
                        'ratio': round(float(ratio), 4),
                        'issue': 'Significant asymmetry detected'
                    })
    
    return {
        'limb_ratios': limb_ratios,
        'asymmetry_issues': issues,
        'passed': len(issues) == 0
    }


def validate_root_position_bounds(skeleton: Skeleton, 
                                   max_vertical_offset: float = 500.0,
                                   max_horizontal_offset: float = 1000.0) -> Dict[str, Any]:
    """Validate that root position stays within reasonable bounds."""
    root_name = 'hips'
    root_bone = skeleton.get_bone_case_insensitive(root_name)
    
    if not root_bone:
        return {
            'passed': False,
            'error': 'Root bone (hips) not found'
        }
    
    positions = []
    if root_bone.world_location_animation:
        positions = list(root_bone.world_location_animation.values())
    elif hasattr(root_bone, 'head'):
        positions = [root_bone.head]
    
    if not positions:
        return {
            'passed': False,
            'error': 'No root position data found'
        }
    
    # Calculate bounds
    positions = np.array(positions)
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    
    vertical_range = abs(max_pos[1] - min_pos[1])  # Y is up in many formats
    horizontal_range = np.sqrt((max_pos[0] - min_pos[0])**2 + (max_pos[2] - min_pos[2])**2)
    
    issues = []
    if vertical_range > max_vertical_offset:
        issues.append(f"Vertical movement ({vertical_range:.2f}) exceeds maximum ({max_vertical_offset})")
    if horizontal_range > max_horizontal_offset:
        issues.append(f"Horizontal movement ({horizontal_range:.2f}) exceeds maximum ({max_horizontal_offset})")
    
    return {
        'vertical_range': round(float(vertical_range), 4),
        'horizontal_range': round(float(horizontal_range), 4),
        'min_position': min_pos.tolist(),
        'max_position': max_pos.tolist(),
        'issues': issues,
        'passed': len(issues) == 0
    }


def validate_skeleton_integrity(input_fbx_path: str, output_fbx_path: str) -> Dict[str, Any]:
    """
    Comprehensive skeleton validation between input and output.
    
    Returns validation results including:
    - Bone count match
    - Bone position comparison
    - Limb length validation
    - Root position bounds
    """
    results = {
        'passed': True,
        'checks': {}
    }
    
    try:
        # Load skeletons
        _, input_scene, input_skel = load_fbx(input_fbx_path)
        _, output_scene, output_skel = load_fbx(output_fbx_path)
        
        # Check 1: Bone count
        input_bones = len(input_skel.bones)
        output_bones = len(output_skel.bones)
        bone_count_match = input_bones == output_bones
        
        results['checks']['bone_count'] = {
            'input': input_bones,
            'output': output_bones,
            'match': bone_count_match,
            'passed': bone_count_match
        }
        
        if not bone_count_match:
            results['passed'] = False
        
        # Check 2: Skeleton height
        input_height = get_skeleton_height(input_skel)
        output_height = get_skeleton_height(output_skel)
        height_ratio = output_height / input_height if input_height > 0 else 0
        height_valid = 0.5 <= height_ratio <= 2.0 and output_height > 10
        
        results['checks']['skeleton_height'] = {
            'input': round(float(input_height), 4),
            'output': round(float(output_height), 4),
            'ratio': round(float(height_ratio), 4),
            'passed': height_valid
        }
        
        if not height_valid:
            results['passed'] = False
        
        # Check 3: Bone positions (if bone count matches)
        if bone_count_match:
            input_positions = get_bone_positions(input_skel)
            output_positions = get_bone_positions(output_skel)
            position_comparison = compare_bone_positions(input_positions, output_positions)
            results['checks']['bone_positions'] = position_comparison
            
            if not position_comparison['passed']:
                results['passed'] = False
        
        # Check 4: Limb lengths
        limb_validation = validate_limb_lengths(output_skel)
        results['checks']['limb_lengths'] = limb_validation
        
        if not limb_validation['passed']:
            results['passed'] = False
        
        # Check 5: Root position bounds
        root_validation = validate_root_position_bounds(output_skel)
        results['checks']['root_position'] = root_validation
        
        if not root_validation['passed']:
            results['passed'] = False
            
    except Exception as e:
        results['passed'] = False
        results['error'] = str(e)
    
    return results
