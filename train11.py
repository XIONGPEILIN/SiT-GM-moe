import torch
import time
import sys

def occupy_vram(device_id=0, target_gb=35):
    """
    Occupies approximately target_gb of VRAM on the specified CUDA device.
    """
    if not torch.cuda.is_available():
        print("Error: CUDA is not available on this system.")
        return

    try:
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
        
        # 1 GB = 1024^3 bytes. float32 takes 4 bytes.
        # num_elements = (GB * 1024^3) / 4
        num_elements = int((target_gb * 1024**3) / 4)
        
        print(f"[*] Attempting to allocate {target_gb} GB on {device}...")
        
        # Use torch.zeros to ensure physical memory is actually touched/paged
        dummy_tensor = torch.zeros(num_elements, dtype=torch.float32, device=device)
        
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        
        print(f"[+] Successfully allocated {allocated:.2f} GB.")
        print(f"[+] Current reserved memory: {reserved:.2f} GB.")
        print("[!] Press Ctrl+C to stop the script and release memory.")
        
        # Keep the process alive
        while True:
            time.sleep(60)
            
    except torch.cuda.OutOfMemoryError:
        print(f"[-] Error: Out of Memory. Cannot allocate {target_gb} GB on {device}.")
        # Suggest available memory
        free, total = torch.cuda.mem_get_info(device)
        print(f"[-] Available: {free/1024**3:.2f} GB / Total: {total/1024**3:.2f} GB")
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user. Releasing VRAM...")
    except Exception as e:
        print(f"[-] An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Default to device 0 and 35GB
    target = 35
    if len(sys.argv) > 1:
        try:
            target = float(sys.argv[1])
        except ValueError:
            pass
            
    occupy_vram(device_id=0, target_gb=target)
