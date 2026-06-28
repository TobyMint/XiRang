"""GPU 显存监控。

优先用 pynvml 读 NVIDIA GPU；不可用时降级为占位实现（返回 0），
保证 proxy 在没有 GPU 的开发机上也能跑起来。
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class GPUSnapshot:
    ts: float
    device: int
    mem_used_mib: float
    mem_total_mib: float
    gpu_util_pct: float

    @property
    def mem_used_pct(self) -> float:
        if self.mem_total_mib <= 0:
            return 0.0
        return 100.0 * self.mem_used_mib / self.mem_total_mib


class GPUMonitor:
    def __init__(self, device: int = 0) -> None:
        self.device = device
        self._pynvml = None
        self._handle = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            self._pynvml = pynvml
        except Exception:
            # 无 GPU / 无 pynvml：静默降级
            self._pynvml = None

    @property
    def available(self) -> bool:
        return self._pynvml is not None

    def snapshot(self) -> GPUSnapshot:
        ts = time.time()
        if not self.available:
            return GPUSnapshot(ts, self.device, 0.0, 0.0, 0.0)
        try:
            info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            return GPUSnapshot(
                ts,
                self.device,
                info.used / (1024 * 1024),
                info.total / (1024 * 1024),
                float(util.gpu),
            )
        except Exception:
            return GPUSnapshot(ts, self.device, 0.0, 0.0, 0.0)

    def close(self) -> None:
        if self.available:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
