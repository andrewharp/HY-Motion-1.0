#!/usr/bin/env python3
"""
HY-Motion-1.0 → Mixamo Retargeting API
FastAPI service for animating GLB avatars from text prompts.
"""
import os
import sys
import base64
import shutil
import tempfile
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager

# Add paths
sys.path.insert(0, '/workspace/HY-Motion-1.0')
sys.path.insert(0, '/workspace/ComfyUI-HyMotion')

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from hymotion_api_pipeline import HYMotionRetargetingPipeline
from validation import validate_skeleton_integrity


# Global pipeline instance
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global pipeline
    # Startup
    print("=" * 60)
    print("Starting HY-Motion Retargeting API")
    print("=" * 60)
    # Use volume-mounted model path (/workspace) for RunPod deployment
    model_path = '/workspace/HY-Motion-1.0/ckpts'
    if not os.path.exists(model_path):
        # Fallback to Docker image path
        model_path = '/app/HY-Motion-1.0/ckpts'
    print(f"Using model path: {model_path}")
    pipeline = HYMotionRetargetingPipeline(model_path=model_path)
    # Model is loaded lazily on first request to avoid startup delays
    print("Pipeline initialized. Model will be loaded on first request.")
    yield
    # Shutdown
    if pipeline:
        pipeline.cleanup()


app = FastAPI(
    title="HY-Motion Retargeting API",
    description="Generate animations for GLB avatars using HY-Motion-1.0",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnimateRequest(BaseModel):
    """Request body for animation generation."""
    prompt: Optional[str] = Field(None, description="Text prompt for motion generation (e.g., 'dancing happily')")
    npz_url: Optional[str] = Field(None, description="URL to NPZ motion file (alternative to prompt)")
    avatar_glb: str = Field(..., description="URL or base64-encoded GLB file")
    output_format: str = Field("glb", description="Output format (glb)")
    num_frames: int = Field(120, ge=30, le=300, description="Number of frames to generate")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")
    yaw_offset: float = Field(0.0, description="Rotation around Y axis in degrees")
    in_place: bool = Field(False, description="Remove horizontal movement")
    preserve_textures: bool = Field(True, description="Preserve original textures")


class AnimateResponse(BaseModel):
    """Response from animation generation."""
    success: bool
    message: str
    validation: dict
    output_url: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    cuda_available: bool
    version: str


# Store for generated files
OUTPUT_DIR = Path("/workspace/api_output")
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API info."""
    import torch
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline is not None and pipeline.runtime is not None,
        cuda_available=torch.cuda.is_available(),
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    import torch
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline is not None and pipeline.runtime is not None,
        cuda_available=torch.cuda.is_available(),
        version="1.0.0"
    )


@app.post("/animate")
async def animate(request: AnimateRequest, background_tasks: BackgroundTasks):
    """
    Generate animation for a GLB avatar from text prompt or NPZ file.
    
    Returns animated GLB with preserved textures.
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Validate request
    if not request.prompt and not request.npz_url:
        raise HTTPException(status_code=400, detail="Either prompt or npz_url must be provided")
    
    if request.output_format != "glb":
        raise HTTPException(status_code=400, detail="Only 'glb' output format is supported")
    
    try:
        # Process the request
        result = pipeline.process(
            avatar_input=request.avatar_glb,
            prompt=request.prompt,
            npz_url=request.npz_url,
            num_frames=request.num_frames,
            seed=request.seed,
            yaw_offset=request.yaw_offset,
            in_place=request.in_place,
            preserve_textures=request.preserve_textures
        )
        
        if not result['success']:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": result.get('error', 'Processing failed'),
                    "validation": result.get('validation', {})
                }
            )
        
        # Move output to persistent location
        output_path = Path(result['output_glb_path'])
        persistent_path = OUTPUT_DIR / f"{output_path.stem}_{os.urandom(4).hex()}.glb"
        shutil.copy(output_path, persistent_path)
        
        # Run additional validation
        validation_result = validate_skeleton_integrity(
            result['validation'].get('input_fbx', ''),
            result['validation'].get('output_fbx', '')
        )
        
        return {
            "success": True,
            "message": "Animation generated successfully",
            "validation": {
                **result['validation'],
                "detailed_checks": validation_result
            },
            "output_url": f"/download/{persistent_path.name}"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/animate-upload")
async def animate_upload(
    avatar_glb: UploadFile = File(..., description="GLB file to animate"),
    prompt: Optional[str] = Form(None, description="Text prompt for motion"),
    npz_url: Optional[str] = Form(None, description="URL to NPZ motion file"),
    num_frames: int = Form(120),
    seed: Optional[int] = Form(None),
    yaw_offset: float = Form(0.0),
    in_place: bool = Form(False),
    preserve_textures: bool = Form(True)
):
    """
    Generate animation by uploading a GLB file directly.
    
    Alternative to /animate for file upload workflows.
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not prompt and not npz_url:
        raise HTTPException(status_code=400, detail="Either prompt or npz_url must be provided")
    
    # Save uploaded file
    temp_dir = tempfile.mkdtemp()
    input_path = Path(temp_dir) / avatar_glb.filename
    
    with open(input_path, 'wb') as f:
        content = await avatar_glb.read()
        f.write(content)
    
    try:
        # Process
        result = pipeline.process(
            avatar_input=str(input_path),
            prompt=prompt,
            npz_url=npz_url,
            num_frames=num_frames,
            seed=seed,
            yaw_offset=yaw_offset,
            in_place=in_place,
            preserve_textures=preserve_textures
        )
        
        # Cleanup temp dir
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if not result['success']:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": result.get('error', 'Processing failed'),
                    "validation": result.get('validation', {})
                }
            )
        
        # Move output to persistent location
        output_path = Path(result['output_glb_path'])
        persistent_path = OUTPUT_DIR / f"{output_path.stem}_{os.urandom(4).hex()}.glb"
        shutil.copy(output_path, persistent_path)
        
        return {
            "success": True,
            "message": "Animation generated successfully",
            "validation": result['validation'],
            "output_url": f"/download/{persistent_path.name}"
        }
        
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download(filename: str):
    """Download generated GLB file."""
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type="model/gltf-binary",
        filename=filename
    )


@app.get("/examples")
async def examples():
    """Get example requests."""
    return {
        "examples": [
            {
                "name": "Basic animation from URL",
                "request": {
                    "prompt": "dancing happily",
                    "avatar_glb": "https://example.com/avatar.glb",
                    "output_format": "glb"
                }
            },
            {
                "name": "Animation from base64",
                "request": {
                    "prompt": "walking forward",
                    "avatar_glb": "data:model/gltf-binary;base64,AAAA...",
                    "num_frames": 120,
                    "seed": 42
                }
            },
            {
                "name": "From NPZ file",
                "request": {
                    "npz_url": "https://example.com/motion.npz",
                    "avatar_glb": "https://example.com/avatar.glb",
                    "output_format": "glb"
                }
            }
        ]
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
