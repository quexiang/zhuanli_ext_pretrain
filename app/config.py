"""Application configuration, resource detection, and parallel planning."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ═══════════════════════════════════════════════════════════════════
# MUST be set BEFORE any code that imports paddle (including the
# ResourceDetector below).  Otherwise paddle reads the flags too late
# and the PIR oneDNN compiler kicks in on Windows.
# ═══════════════════════════════════════════════════════════════════
for _flag in ("FLAGS_enable_pir_api", "FLAGS_use_onednn_op", "FLAGS_use_onednn_graph"):
    os.environ.setdefault(_flag, "0")

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# App settings (unchanged)
# ═══════════════════════════════════════════════════════════════════
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    app_name: str = "专利说明书文本提取工具"
    max_file_size: int = 500 * 1024 * 1024  # 500MB
    upload_dir: Path = Path("uploads")
    output_dir: Path = Path("outputs")
    ocr_dpi: int = 200
    segment_min_len: int = 50  # Note: process_document now defaults to 120
    segment_max_len: int = 2000
    category: str = "专利文献"
    standard_chunk_size: int = 150


settings = Settings()

# Ensure directories exist
settings.upload_dir.mkdir(exist_ok=True)
settings.output_dir.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Resource & parallel configuration data classes
# ═══════════════════════════════════════════════════════════════════
@dataclass
class ResourceInfo:
    """Runtime-detected system resources."""

    cpu_count: int = 1
    total_ram_gb: float = 1.0
    available_ram_gb: float = 1.0
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_vram_mb: int = 0
    paddle_backend: str = "cpu"  # "cpu" | "gpu" | "mps"

    def __str__(self) -> str:
        gpu_str = (
            f"{self.gpu_count}×GPU ({self.gpu_vram_mb}MB)"
            if self.gpu_available
            else "none"
        )
        return (
            f"ResourceInfo(cpu={self.cpu_count}, ram={self.available_ram_gb:.1f}/{self.total_ram_gb:.1f}GB, "
            f"gpu={gpu_str}, paddle={self.paddle_backend})"
        )


@dataclass
class ParallelConfig:
    """Optimal parallelism configuration computed from resources."""

    max_workers: int = 1
    batch_size: int = 1
    strategy: str = "serial"  # "cpu_multiprocess" | "gpu_batch" | "serial"

    # ── Standard-specific thresholds (lighter than patent OCR) ──────
    standard_min_workers: int = 1  # minimum for standards processing
    standard_ram_per_worker_gb: float = 0.3  # ~300 MB per standard worker
    standard_max_workers: int = 16  # higher cap for standards

    def __str__(self) -> str:
        return (
            f"ParallelConfig(strategy={self.strategy}, workers={self.max_workers}, "
            f"batch={self.batch_size})"
        )


# ═══════════════════════════════════════════════════════════════════
# System resource detector
# ═══════════════════════════════════════════════════════════════════
class ResourceDetector:
    """Detect available system resources at runtime.

    Checks CPU cores, RAM, GPU (CUDA / ROCm / MPS), and PaddlePaddle backend.
    All methods are static; call ``ResourceDetector.detect()`` once at startup.
    """

    @staticmethod
    def _get_ram_gb() -> tuple[float, float]:
        """Return ``(total_gb, available_gb)`` or a safe estimate."""
        # 1) psutil (cross-platform, best)
        try:
            import psutil

            mem = psutil.virtual_memory()
            return mem.total / 1e9, mem.available / 1e9
        except ImportError:
            pass

        # 2) sysctl (macOS)
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                total_bytes = int(result.stdout.strip().split()[1])
                # macOS doesn't report "available" easily; assume 70% free
                return total_bytes / 1e9, total_bytes * 0.7 / 1e9
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
            pass

        # 3) /proc/meminfo (Linux containers)
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

        # 4) Fallback: assume 1 GB (safe minimal)
        return 1.0, 1.0

    @staticmethod
    def _check_nvidia_gpu() -> tuple[int, int]:
        """Return ``(count, vram_mb_per_gpu)`` via ``nvidia-smi``."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=count,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().splitlines()
                total_vram = 0
                for line in lines:
                    parts = line.split(", ")
                    if len(parts) >= 2:
                        total_vram += int(parts[1].strip())
                count = len(lines)
                avg_vram = total_vram // count if count else 0
                return count, avg_vram
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return 0, 0

    @staticmethod
    def _check_gpu_paddle() -> tuple[bool, str]:
        """Check if PaddlePaddle detects GPU support."""
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
        """Detect all system resources and return a ``ResourceInfo``."""
        cpu_count = os.cpu_count() or 1
        total_ram, avail_ram = ResourceDetector._get_ram_gb()

        # GPU detection
        gpu_available_paddle, backend = ResourceDetector._check_gpu_paddle()
        gpu_count, gpu_vram = (0, 0)
        if gpu_available_paddle:
            gpu_count, gpu_vram = ResourceDetector._check_nvidia_gpu()
            if gpu_count == 0:
                gpu_count = 1  # Paddle says GPU is available but nvidia-smi failed
                gpu_vram = 8192  # assume 8GB default

        gpu_available = gpu_count > 0

        return ResourceInfo(
            cpu_count=cpu_count,
            total_ram_gb=total_ram,
            available_ram_gb=avail_ram,
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            gpu_vram_mb=gpu_vram,
            paddle_backend=backend,
        )


# ═══════════════════════════════════════════════════════════════════
# Parallel strategy planner
# ═══════════════════════════════════════════════════════════════════
class ParallelPlanner:
    """Compute the optimal parallelism configuration from system resources."""

    # Conservative RAM estimate per PaddleOCR worker instance (GB)
    _OCR_RAM_PER_WORKER_GB = 1.8

    # Reserve CPU cores for system / I/O
    _CPU_RESERVE = 2

    # Absolute upper limit on CPU workers (diminishing returns beyond this)
    _MAX_CPU_WORKERS = 8

    @staticmethod
    def plan(info: ResourceInfo) -> ParallelConfig:
        """Select optimal parallelism based on detected resources.

        Strategy selection
        -------------------
        * **gpu_batch**: GPU is available (CUDA).  Use 1-2 workers (one per GPU),
          large image batches per ``predict()`` call.  GPU inference is fast,
          so few workers are needed.

        * **cpu_multiprocess**: No GPU.  Use N workers based on CPU count and
          available RAM.  Each worker is a separate process with its own
          PaddleOCR instance.  Pages within a PDF are OCR'd sequentially in
          that worker.

        * **serial**: Very limited resources (≤ 2 cores or ≤ 4 GB RAM).
          Single-process sequential processing.
        """
        # ── GPU path ──────────────────────────────────────────────
        if info.gpu_available:
            vram_mb = max(info.gpu_vram_mb, 4096)  # at least 4 GB
            # Larger VRAM → larger batches
            batch_size = min(max(4, vram_mb // 2048), 16)
            workers = min(info.gpu_count, 2)
            return ParallelConfig(
                max_workers=workers,
                batch_size=batch_size,
                strategy="gpu_batch",
            )

        # ── CPU path ──────────────────────────────────────────────
        cpu_based = max(1, info.cpu_count - ParallelPlanner._CPU_RESERVE)
        mem_based = max(1, int(info.available_ram_gb / ParallelPlanner._OCR_RAM_PER_WORKER_GB))
        workers = min(cpu_based, mem_based)
        workers = max(1, workers)

        # Fall back to serial if resources are very limited
        if workers <= 1 or info.cpu_count <= 2 or info.available_ram_gb < 4:
            return ParallelConfig(
                max_workers=1,
                batch_size=1,
                strategy="serial",
            )

        # Cap to prevent diminishing returns
        workers = min(workers, ParallelPlanner._MAX_CPU_WORKERS)

        return ParallelConfig(
            max_workers=workers,
            batch_size=1,
            strategy="cpu_multiprocess",
        )

    @staticmethod
    def plan_standard(info: ResourceInfo) -> ParallelConfig:
        """Compute optimal parallelism for standard-document processing.

        Standards processing is lighter than patent OCR (no PaddleOCR model),
        so we use lower RAM-per-worker estimate and higher worker cap.
        Returns a new ParallelConfig with adjusted standard-specific fields.
        """
        config = ParallelPlanner.plan(info)  # start with base config

        if info.gpu_available:
            vram_mb = max(info.gpu_vram_mb, 4096)
            batch_size = min(max(4, vram_mb // 2048), 16)
            workers = min(info.gpu_count, 2)
            config.max_workers = workers
            config.batch_size = batch_size
            config.strategy = "gpu_batch"
            config.standard_max_workers = max(workers * 4, 8)
            config.standard_min_workers = workers
            return config

        # CPU path: standards are lightweight, allow more workers
        cpu_based = max(2, info.cpu_count - 1)  # reserve only 1 core
        mem_based = max(2, int(info.available_ram_gb / config.standard_ram_per_worker_gb))
        workers = min(cpu_based, mem_based)
        workers = max(config.standard_min_workers, workers)
        workers = min(workers, config.standard_max_workers)

        if workers <= 1 or info.cpu_count <= 2 or info.available_ram_gb < 2:
            workers = 1

        config.max_workers = max(config.max_workers, workers)
        config.standard_min_workers = max(config.standard_min_workers, 1)
        config.standard_max_workers = max(config.standard_max_workers, workers)
        config.strategy = "cpu_multiprocess"

        return config


# ═══════════════════════════════════════════════════════════════════
# Global singleton: detected once at import time
# ═══════════════════════════════════════════════════════════════════
resource_info = ResourceDetector.detect()
parallel_config = ParallelPlanner.plan_standard(resource_info)

logger.info("System resources: %s", resource_info)
logger.info("Parallel config: %s", parallel_config)
