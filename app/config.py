"""Application configuration, resource detection, and parallel planning."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# App settings
# ═══════════════════════════════════════════════════════════════════
class Settings(BaseSettings):
    app_name: str = "说明书文本提取工具"
    max_file_size: int = 500 * 1024 * 1024
    upload_dir: Path = Path("uploads")
    output_dir: Path = Path("outputs")
    ocr_dpi: int = 300
    segment_min_len: int = 50
    segment_max_len: int = 2000
    mode: str = "patent"

    # 文本质量检查参数（供标准模式使用）
    min_valid_text_length: int = 100        # 直接提取文本的最短有效长度
    cjk_ratio_threshold: float = 0.05       # CJK 占比阈值（5%）

    @property
    def category(self) -> str:
        return "标准文献" if self.mode == "standard" else "专利文献"

    class Config:
        env_file = ".env"


settings = Settings()

# Ensure directories exist
settings.upload_dir.mkdir(exist_ok=True)
settings.output_dir.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Resource & parallel configuration (保持原有不变)
# ═══════════════════════════════════════════════════════════════════
@dataclass
class ResourceInfo:
    cpu_count: int = 1
    total_ram_gb: float = 1.0
    available_ram_gb: float = 1.0
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_vram_mb: int = 0
    paddle_backend: str = "cpu"

    def __str__(self) -> str:
        gpu_str = f"{self.gpu_count}×GPU ({self.gpu_vram_mb}MB)" if self.gpu_available else "none"
        return (
            f"ResourceInfo(cpu={self.cpu_count}, ram={self.available_ram_gb:.1f}/{self.total_ram_gb:.1f}GB, "
            f"gpu={gpu_str}, paddle={self.paddle_backend})"
        )


@dataclass
class ParallelConfig:
    max_workers: int = 1
    batch_size: int = 1
    strategy: str = "serial"

    def __str__(self) -> str:
        return f"ParallelConfig(strategy={self.strategy}, workers={self.max_workers}, batch={self.batch_size})"


class ResourceDetector:
    @staticmethod
    def _get_ram_gb() -> tuple[float, float]:
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.total / 1e9, mem.available / 1e9
        except ImportError:
            pass
        try:
            import subprocess
            result = subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                total_bytes = int(result.stdout.strip().split()[1])
                return total_bytes / 1e9, total_bytes * 0.7 / 1e9
        except Exception:
            pass
        try:
            with open("/proc/meminfo") as f:
                data = f.read()
            total_kb = 0
            avail_kb = 0
            for line in data.splitlines():
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
            if total_kb:
                return total_kb / 1e6, (avail_kb or total_kb) / 1e6
        except OSError:
            pass
        return 1.0, 1.0

    @staticmethod
    def _check_nvidia_gpu() -> tuple[int, int]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().splitlines()
                total_vram = 0
                for line in lines:
                    parts = line.split(", ")
                    if len(parts) >= 2:
                        total_vram += int(parts[1].strip())
                return len(lines), total_vram // len(lines) if len(lines) else 0
        except Exception:
            pass
        return 0, 0

    @staticmethod
    def _check_gpu_paddle() -> tuple[bool, str]:
        try:
            import paddle
            if paddle.is_compiled_with_cuda():
                return True, "gpu"
            device = paddle.device.get_device()
            if "gpu" in device:
                return True, "gpu"
            return False, "cpu"
        except ImportError:
            return False, "cpu"
        except Exception:
            return False, "cpu"

    @staticmethod
    def detect() -> ResourceInfo:
        cpu_count = os.cpu_count() or 1
        total_ram, avail_ram = ResourceDetector._get_ram_gb()
        gpu_available_paddle, backend = ResourceDetector._check_gpu_paddle()
        gpu_count, gpu_vram = (0, 0)
        if gpu_available_paddle:
            gpu_count, gpu_vram = ResourceDetector._check_nvidia_gpu()
            if gpu_count == 0:
                gpu_count = 1
                gpu_vram = 8192
        return ResourceInfo(
            cpu_count=cpu_count,
            total_ram_gb=total_ram,
            available_ram_gb=avail_ram,
            gpu_available=gpu_count > 0,
            gpu_count=gpu_count,
            gpu_vram_mb=gpu_vram,
            paddle_backend=backend,
        )


class ParallelPlanner:
    _OCR_RAM_PER_WORKER_GB = 1.8
    _CPU_RESERVE = 2
    _MAX_CPU_WORKERS = 8

    @staticmethod
    def plan(info: ResourceInfo) -> ParallelConfig:
        if info.gpu_available:
            vram_mb = max(info.gpu_vram_mb, 4096)
            batch_size = min(max(4, vram_mb // 2048), 16)
            return ParallelConfig(max_workers=min(info.gpu_count, 2), batch_size=batch_size, strategy="gpu_batch")

        cpu_based = max(1, info.cpu_count - ParallelPlanner._CPU_RESERVE)
        mem_based = max(1, int(info.available_ram_gb / ParallelPlanner._OCR_RAM_PER_WORKER_GB))
        workers = min(cpu_based, mem_based)
        workers = max(1, workers)

        if workers <= 1 or info.cpu_count <= 2 or info.available_ram_gb < 4:
            return ParallelConfig(max_workers=1, batch_size=1, strategy="serial")

        workers = min(workers, ParallelPlanner._MAX_CPU_WORKERS)
        return ParallelConfig(max_workers=workers, batch_size=1, strategy="cpu_multiprocess")


resource_info = ResourceDetector.detect()
parallel_config = ParallelPlanner.plan(resource_info)

logger.info("System resources: %s", resource_info)
logger.info("Parallel config: %s", parallel_config)