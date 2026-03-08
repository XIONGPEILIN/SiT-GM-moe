import os
import torch

def test_compile_paths():
    print(f"TMPDIR: {os.environ.get('TMPDIR', 'not set')}")
    print(f"TORCHINDUCTOR_CACHE_DIR: {os.environ.get('TORCHINDUCTOR_CACHE_DIR', 'not set')}")
    print(f"TRITON_CACHE_DIR: {os.environ.get('TRITON_CACHE_DIR', 'not set')}")

    try:
        import torch._inductor.config as inductor_config
        print(f"Inductor cache_dir config: {inductor_config.cache_dir}")
    except:
        print("Could not access inductor_config.cache_dir")

    try:
        import triton
        # Triton doesn't have a simple config for cache dir in python easily accessible,
        # but it uses TRITON_CACHE_DIR env var.
    except:
        print("Triton not found")

if __name__ == "__main__":
    test_compile_paths()
