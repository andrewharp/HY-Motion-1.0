#!/usr/bin/env python3
"""
Patch for ComfyUI-HyMotion text encoder to fix meta tensor issues.
Run this after cloning ComfyUI-HyMotion.
"""

import os
import sys

FILE_PATH = "/app/ComfyUI-HyMotion/hymotion/network/text_encoders/text_encoder.py"

def patch_text_encoder():
    with open(FILE_PATH, 'r') as f:
        content = f.read()
    
    # Find the line with llm_text_encoder.eval() and add materialization after it
    target_line = "self.llm_text_encoder.eval().requires_grad_(False)"
    
    if target_line not in content:
        print(f"Target line not found in {FILE_PATH}")
        sys.exit(1)
    
    if "Materialize any remaining meta tensors" in content:
        print("Patch already applied")
        sys.exit(0)
    
    patch_code = '''        self.llm_text_encoder.eval().requires_grad_(False)
        
        # Materialize any remaining meta tensors for API
        for model in [self.sentence_emb_text_encoder, self.llm_text_encoder]:
            for name, param in model.named_parameters():
                if param.device.type == "meta":
                    print(f"[HYTextModel] Materializing meta tensor: {name}")
                    from accelerate.utils import set_module_tensor_to_device
                    set_module_tensor_to_device(model, name, "cpu", torch.empty_like(param, device="cpu"))'''
    
    content = content.replace(target_line, patch_code)
    
    with open(FILE_PATH, 'w') as f:
        f.write(content)
    
    print(f"Patched {FILE_PATH} successfully")

if __name__ == "__main__":
    patch_text_encoder()
