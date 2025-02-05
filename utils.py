import sys

import torch
import triton


def get_hardware_info() -> dict[str, str | int]:
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        "count": torch.cuda.device_count() if torch.cuda.is_available() else 1,
    }


def get_software_info() -> dict[str, str]:
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch_version": torch.__version__.split("+")[0],
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "triton_version": triton.__version__,
    }
