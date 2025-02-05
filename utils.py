import sys
import torch
import triton

def get_hardware_info() -> dict[str, str | int]:
    return {
        "hardware.device": "cuda" if torch.cuda.is_available() else "cpu",
        "hardware.name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        "hardware.count": torch.cuda.device_count() if torch.cuda.is_available() else 1,
    }

def get_software_info() -> dict[str, str]:
    return {
        "software.python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "software.torch_version": torch.__version__,
        "software.cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "software.triton_version": triton.__version__,
    }
