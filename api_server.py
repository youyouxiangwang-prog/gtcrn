"""
GTCRN Speech Enhancement API Server
Takes PCM audio input, returns denoised PCM audio output
"""
import os
import io
import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn

from gtcrn import GTCRN

app = FastAPI(title="GTCRN Speech Enhancement API")

# Model and device
device = torch.device("cpu")
model = None

# Audio parameters
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512


def load_model():
    """Load the GTCRN model"""
    global model
    model = GTCRN().eval()
    ckpt_path = os.path.join('checkpoints', 'model_trained_on_dns3.tar')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"Model loaded from {ckpt_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
    return model


def denoise_audio(audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Denoise audio using GTCRN
    
    Args:
        audio_data: Input audio as numpy array (float32)
        sample_rate: Sample rate (expected 16000 Hz)
    
    Returns:
        Denoised audio as numpy array
    """
    if model is None:
        load_model()
    
    # Ensure correct sample rate
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected sample rate {SAMPLE_RATE} Hz, got {sample_rate}")
    
    # Convert to tensor and ensure 1D
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)  # Convert stereo to mono
    
    # STFT - use return_complex=True for PyTorch compatibility
    input_spec = torch.stft(
        torch.from_numpy(audio_data),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=torch.hann_window(WIN_LENGTH).pow(0.5),
        return_complex=True
    )
    
    # Convert complex to [real, imag] format for model
    input_spec_real = torch.view_as_real(input_spec)  # [F, T, 2]
    
    # Inference and ISTFT - all in no_grad to avoid gradient tracking
    with torch.no_grad():
        # Model expects [B, F, T, 2] and returns [B, F, T, 2]
        output_spec = model(input_spec_real.unsqueeze(0))[0]  # [F, T, 2]
        
        # Convert back to complex (need contiguous tensor)
        output_spec_complex = torch.view_as_complex(output_spec.contiguous())  # [F, T]
        
        # ISTFT
        enhanced = torch.istft(
            output_spec_complex,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=torch.hann_window(WIN_LENGTH).pow(0.5)
        )
    
    return enhanced.detach().cpu().numpy()


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "GTCRN"}


class DenoiseRequest(BaseModel):
    """JSON request for S3-based denoising"""
    input_s3_uri: str
    output_s3_uri: str


@app.post("/denoise/")
async def denoise(file: UploadFile = File(...)):
    """
    Denoise audio file (PCM)
    
    Accepts: PCM audio file (16-bit, 16kHz, mono)
    Returns: Denoised PCM audio file
    """
    if model is None:
        load_model()
    
    # Read uploaded file
    content = await file.read()
    
    try:
        # Try to read as audio file
        audio, sr = sf.read(io.BytesIO(content), dtype='float32')
    except Exception as e:
        # If soundfile fails, assume raw PCM
        try:
            # Interpret as 16-bit PCM at 16kHz
            pcm_data = np.frombuffer(content, dtype=np.int16)
            audio = pcm_data.astype(np.float32) / 32768.0
            sr = SAMPLE_RATE
        except Exception as e2:
            raise HTTPException(status_code=400, detail=f"Cannot parse audio: {e2}")
    
    # Denoise
    enhanced = denoise_audio(audio, sr)
    
    # Convert back to 16-bit PCM
    enhanced_int16 = (enhanced * 32767).astype(np.int16)
    
    # Return as binary PCM
    return Response(
        content=enhanced_int16.tobytes(),
        media_type="audio/pcm",
        headers={"Content-Disposition": f"attachment; filename=denoised.pcm"}
    )


@app.post("/denoise/s3")
async def denoise_s3(request: DenoiseRequest):
    """
    Denoise audio from S3 URI
    
    Request body:
    {
        "input_s3_uri": "s3://bucket/path/to/input.pcm",
        "output_s3_uri": "s3://bucket/path/to/output.pcm"
    }
    """
    # For now, return a simple acknowledgment
    # Full S3 support requires boto3 and proper credentials
    return {
        "status": "received",
        "input_s3_uri": request.input_s3_uri,
        "output_s3_uri": request.output_s3_uri,
        "message": "S3-based denoising - configure boto3 for full support"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
