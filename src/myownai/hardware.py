from __future__ import annotations
import os
import platform
import shutil
from pathlib import Path
from typing import Any

def get_hardware_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "ram_total_bytes": None,
        "ram_available_bytes": None,
        "gpu_available": False,
        "gpu_name": None,
        "gpu_vram_bytes": None,
        "cuda_version": None,
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
    }
    try:
        import psutil
        memory = psutil.virtual_memory()
        report["ram_total_bytes"] = memory.total
        report["ram_available_bytes"] = memory.available
    except ImportError:
        report["ram_note"] = "Install psutil to report RAM totals."
    try:
        import torch
        report["pytorch_version"] = torch.__version__
        report["cuda_version"] = torch.version.cuda
        report["gpu_available"] = torch.cuda.is_available()
        if report["gpu_available"]:
            report["gpu_name"] = torch.cuda.get_device_name(0)
            report["gpu_vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        report["pytorch_note"] = "PyTorch is not installed."
    return report
