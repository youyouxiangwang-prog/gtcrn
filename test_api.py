"""
GTCRN API Test Suite
Tests for the denoising API endpoints
"""
import requests
import numpy as np
import soundfile as sf
import io
import os

BASE_URL = "http://100.51.85.99:8005"

def create_test_pcm(filename="test_input.pcm", duration_sec=3, sample_rate=16000):
    """Create a test PCM file with some audio-like signal"""
    # Generate a test signal (sine wave mixed with noise)
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate))
    
    # Clean signal: 440Hz sine wave (A note)
    clean = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, len(t))
    noisy = clean + noise
    
    # Convert to 16-bit PCM
    pcm_data = (noisy * 32767).astype(np.int16)
    
    with open(filename, 'wb') as f:
        f.write(pcm_data.tobytes())
    
    print(f"Created test file: {filename} ({len(pcm_data)} samples, {duration_sec} sec)")
    return filename


def test_health():
    """Test health endpoint"""
    print("\n=== Test: Health Check ===")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        print(f"✓ Health check passed: {resp.json()}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_denoise_pcm():
    """Test denoise endpoint with PCM file"""
    print("\n=== Test: Denoise PCM ===")
    
    # Create test file
    test_file = create_test_pcm()
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': ('test.pcm', f, 'audio/pcm')}
            resp = requests.post(f"{BASE_URL}/denoise/", files=files, timeout=30)
        
        resp.raise_for_status()
        
        # Check response
        content_type = resp.headers.get('Content-Type', '')
        print(f"✓ Denoise request succeeded")
        print(f"  Content-Type: {content_type}")
        print(f"  Response size: {len(resp.content)} bytes")
        
        # Save output
        output_file = 'test_output.pcm'
        with open(output_file, 'wb') as f:
            f.write(resp.content)
        print(f"  Output saved to: {output_file}")
        
        # Verify output is valid PCM
        output_data = np.frombuffer(resp.content, dtype=np.int16)
        print(f"  Output samples: {len(output_data)}")
        
        # Clean up
        os.remove(test_file)
        
        return True
    except Exception as e:
        print(f"✗ Denoise failed: {e}")
        return False


def test_denoise_wav():
    """Test denoise endpoint with WAV file"""
    print("\n=== Test: Denoise WAV ===")
    
    # Create a test WAV file
    test_file = "test_input.wav"
    sample_rate = 16000
    duration = 3
    
    t = np.linspace(0, duration, int(duration * sample_rate))
    clean = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = np.random.normal(0, 0.1, len(t))
    audio = clean + noise
    
    sf.write(test_file, audio.astype(np.float32), sample_rate)
    print(f"Created test WAV: {test_file}")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': ('test.wav', f, 'audio/wav')}
            resp = requests.post(f"{BASE_URL}/denoise/", files=files, timeout=30)
        
        resp.raise_for_status()
        print(f"✓ WAV denoise succeeded, output: {len(resp.content)} bytes")
        
        # Save and verify
        output_file = 'test_output.wav'
        with open(output_file, 'wb') as f:
            f.write(resp.content)
        print(f"  Output saved to: {output_file}")
        
        os.remove(test_file)
        return True
    except Exception as e:
        print(f"✗ WAV denoise failed: {e}")
        return False


def test_s3_endpoint():
    """Test S3 denoise endpoint"""
    print("\n=== Test: S3 Endpoint ===")
    try:
        payload = {
            "input_s3_uri": "s3://capsoul-audio-useast/denoise/yangpin.pcm",
            "output_s3_uri": "s3://capsoul-audio-useast/denoise/yangpin_denoised.pcm"
        }
        resp = requests.post(f"{BASE_URL}/denoise/s3", json=payload, timeout=5)
        resp.raise_for_status()
        print(f"✓ S3 endpoint: {resp.json()}")
        return True
    except Exception as e:
        print(f"✗ S3 endpoint failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("GTCRN API Test Suite")
    print("=" * 50)
    
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Denoise PCM", test_denoise_pcm()))
    results.append(("Denoise WAV", test_denoise_wav()))
    results.append(("S3 Endpoint", test_s3_endpoint()))
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed.")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
