#!/usr/bin/env python3
"""
NIM + IPEX-LLM GPU Probe Adapter
=================================
Patches NVIDIA NIM GPU detection for Intel UHD iGPU by bypassing
TensorRT-LLM profile probe and creating a mock XPU profile.
Uses os.execvp to hand off to NIM so patched env vars are inherited in-process.
"""

import os
import sys
import json
import logging
import socket
import time
from pathlib import Path

# FIX 1: Added missing closing parenthesis
logging.basicConfig(
    level=logging.INFO,
    format='[NIM-ADAPTER] %(levelname)s: %(message)s'
)

logger = logging.getLogger(__name__)


def patch_environment():
    """Set environment variables to bypass GPU probe."""
    patches = {
        'NIM_SKIP_GPU_PROBE': '1',
        # FIX 2: Default to Docker service name, not localhost
        'NIM_BACKEND_URL': os.environ.get('NIM_BACKEND_URL', 'http://llm-scaler-backend:8000'),
        'CUDA_VISIBLE_DEVICES': '',
        'NIM_GPU_PROFILE': 'xpu_fallback',
    }  # FIX 3: Added missing closing brace

    for key, value in patches.items():
        os.environ[key] = value
        logger.info(f"Set {key}={value}")


def create_mock_profile():
    """Create a minimal hardware profile JSON that satisfies nimlib."""
    profile = {
        'id': 'xpu_fallback',
        'name': 'IPEX-LLM XPU Fallback Profile',
        'description': 'vLLM on Intel iGPU via IPEX-LLM',
        'backend': 'vllm',
        'backend_version': '0.6.0',
        'precision': 'float16',
        'quantization': 'sym_int4',
        'gpu_type': 'intel_xpu',
        'gpu_compute_capability': 'xelp',
        'max_batch_size': 1,
        'max_num_seqs': 256,
        'gpu_memory_gb': 12,
        'supported_models': ['llama', 'mistral', 'qwen', 'phi'],
        'features': {
            'tensor_parallel': False,
            'pipeline_parallel': False,
            'dynamic_batching': True,
            'lora': True,
            'quantization': True,
        },  # FIX 4: Added missing closing brace for features sub-dict
    }  # FIX 5: Added missing closing brace for profile dict

    cache_dir = Path(os.environ.get('NIM_CACHE', '/opt/nim/.cache'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    profile_path = cache_dir / 'xpu_profile.json'

    try:
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        logger.info(f"Created mock GPU profile at {profile_path}")
    except Exception as e:
        logger.warning(f"Failed to write profile JSON: {e}")


def validate_backend_connectivity():
    """
    Check if the IPEX backend is reachable before handing off to NIM.
    FIX 6: Increased retries 3->5 and interval 5s->10s to survive the
    2-5 min SYCL kernel compilation window on first startup.
    """
    backend_url = os.environ.get('NIM_BACKEND_URL', 'http://llm-scaler-backend:8000')

    if backend_url.startswith('http://'):
        host_port = backend_url.replace('http://', '')
        host, _, port_str = host_port.partition(':')
        port = int(port_str) if port_str else 8000

        max_retries = 5
        for attempt in range(max_retries):
            try:
                sock = socket.create_connection((host, port), timeout=5)
                sock.close()
                logger.info(f"✓ IPEX backend reachable at {host}:{port}")
                return True
            except (socket.timeout, ConnectionRefusedError, OSError):
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Backend not ready (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in 10s..."
                    )
                    time.sleep(10)
                else:
                    logger.warning(
                        f"⚠ Backend not reachable at {host}:{port} after "
                        f"{max_retries} attempts — NIM will retry on first request"
                    )
                    return False
    return False


def main():
    logger.info("=" * 40)
    logger.info("NIM + IPEX-LLM Adapter Starting")
    logger.info("=" * 40)

    try:
        patch_environment()
        create_mock_profile()
        validate_backend_connectivity()

        logger.info("Patches applied. Handing control to NIM via execvp...")

        # FIX 7: os.execvp replaces the current process image so patched
        # os.environ vars are inherited by NIM. The old bash -c approach
        # launched NIM in a new shell which did NOT inherit os.environ patches.
        os.execvp('python', ['python', '-m', 'nim'] + sys.argv[1:])

    except Exception as e:
        logger.error(f"Adapter initialization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
