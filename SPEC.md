# GTCRN Containerization Specification

## Project Overview
- **Name**: GTCRN Speech Enhancement API
- **Type**: Docker containerized inference API
- **Core Functionality**: Speech enhancement (denoising) using GTCRN model - takes audio input, outputs enhanced audio
- **Target Users**: Developers integrating speech enhancement into their applications

## Functionality Specification

### Core Features
1. **REST API Server**
   - FastAPI-based HTTP server
   - Single endpoint for denoising: `POST /denoise/`
   - Health check endpoint: `GET /health`

2. **Input/Output**
   - **Input**: PCM audio file (16-bit, 16kHz, mono)
   - **Output**: PCM audio file (same format - denoised)
   - Support for both binary upload and S3 URI references

3. **API Endpoints**
   ```
   POST /denoise/
   - Content-Type: multipart/form-data or application/json
   - Body (form): file = PCM binary data
   - OR Body (json): {"input_s3_uri": "s3://...", "output_s3_uri": "s3://..."}
   
   Response: Enhanced audio as binary or S3 output confirmation
   ```

4. **Model Loading**
   - Load pre-trained GTCRN model from `checkpoints/` directory
   - Device: CPU (primary), GPU if available

### Data Flow
1. Client sends PCM audio (16kHz, 16-bit mono)
2. Server reads PCM, converts to tensor
3. Apply STFT (512 FFT, 256 hop, 512 window)
4. Run GTCRN inference
5. Apply ISTFT to get enhanced audio
6. Return enhanced PCM

### Configuration
- Sample rate: 16000 Hz
- FFT size: 512
- Hop length: 256
- Window: Hann window
- Bit depth: 16-bit PCM

## Technical Stack
- Python 3.9+
- PyTorch 1.11+
- FastAPI
- Uvicorn
- soundfile (for audio I/O)
- boto3 (for S3 support)

## Docker Configuration
- Base image: `python:3.9-slim`
- PyTorch CPU build for smaller image
- Port: 8005
- Health check endpoint

## Acceptance Criteria
1. API starts and responds to `/health`
2. PCM audio input returns denoised PCM output
3. Denoised audio has same sample rate and format as input
4. Docker image builds successfully
5. Container runs and processes audio correctly
