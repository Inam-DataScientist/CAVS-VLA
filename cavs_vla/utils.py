"""Seeding, logging, timing, hashing and environment capture."""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import numpy as np

_LOGGERS: Dict[str, logging.Logger] = {}


def get_logger(name: str = "cavs", log_file: Optional[str] = None) -> logging.Logger:
    key = f"{name}:{log_file}"
    if key in _LOGGERS:
        return _LOGGERS[key]
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    _LOGGERS[key] = logger
    return logger


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch = _optional_torch()
    if torch is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def _optional_torch():
    """torch if installed, else None (the data builders and HJ solver run without it)."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash01(text: str, seed: int = 0) -> float:
    """Deterministic value in [0, 1) used for split assignment (independent of PYTHONHASHSEED)."""
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "little") / float(1 << 64)


def git_sha(cwd: Optional[str] = None) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd or os.getcwd(),
                             capture_output=True, text=True, timeout=5)
        sha = out.stdout.strip()
        if out.returncode != 0 or not sha:
            return "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd or os.getcwd(),
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_sha": git_sha(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        info["torch"] = None
    return info


class LatencyMeter:
    """Accumulates wall-clock latency per named stage (CUDA-synchronised when on GPU)."""

    def __init__(self, cuda_sync: bool = True) -> None:
        self.samples: Dict[str, list] = {}
        self.cuda_sync = cuda_sync

    def _sync(self) -> None:
        if not self.cuda_sync:
            return
        torch = _optional_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.samples.setdefault(name, []).append((time.perf_counter() - t0) * 1000.0)

    def summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for k, v in self.samples.items():
            arr = np.asarray(v, dtype=np.float64)
            out[k] = {"mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)),
                      "p95_ms": float(np.percentile(arr, 95)), "max_ms": float(arr.max()), "n": int(arr.size)}
        return out


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
