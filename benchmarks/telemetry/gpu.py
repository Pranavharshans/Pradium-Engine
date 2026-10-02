"""Generic NVIDIA GPU telemetry.

Prefers NVML Python bindings when available and falls back to a low-frequency
``nvidia-smi`` query. When neither is available, GPU metrics are unavailable
(recorded as ``None`` / ``UNAVAILABLE`` provenance) — never fabricated.

Units: VRAM in MiB, power in watts, temperature in Celsius, clocks in MHz,
PCIe throughput in MB/s. NVML power is GPU-side power, not whole-system power.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GpuSample:
    """One GPU telemetry sample."""

    timestamp_ns: int
    util_gpu_pct: float | None = None
    util_mem_pct: float | None = None
    power_w: float | None = None
    temperature_c: float | None = None
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    clock_mhz: float | None = None
    mem_clock_mhz: float | None = None
    pcie_rx_mb_s: float | None = None
    pcie_tx_mb_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "util_gpu_pct": self.util_gpu_pct,
            "util_mem_pct": self.util_mem_pct,
            "power_w": self.power_w,
            "temperature_c": self.temperature_c,
            "vram_used_mb": self.vram_used_mb,
            "vram_total_mb": self.vram_total_mb,
            "clock_mhz": self.clock_mhz,
            "mem_clock_mhz": self.mem_clock_mhz,
            "pcie_rx_mb_s": self.pcie_rx_mb_s,
            "pcie_tx_mb_s": self.pcie_tx_mb_s,
        }


def probe_gpus() -> list[dict[str, Any]]:
    """Detect NVIDIA GPUs without assuming any particular model."""
    gpus = _probe_gpus_nvml()
    if gpus:
        return gpus
    return _probe_gpus_smi()


def _probe_gpus_nvml() -> list[dict[str, Any]]:
    try:
        import pynvml  # type: ignore
    except Exception:
        return []
    try:
        pynvml.nvmlInit()
    except Exception:
        return []
    gpus: list[dict[str, Any]] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8", "replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            driver = None
            try:
                driver = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver, bytes):
                    driver = driver.decode("utf-8", "replace")
            except Exception:
                pass
            cuda_version = None
            try:
                raw = pynvml.nvmlSystemGetCudaDriverVersion_v2()
                cuda_version = f"{raw // 1000}.{(raw % 1000) // 10}"
            except Exception:
                pass
            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "uuid": uuid,
                    "vram_total_mb": round(mem.total / (1024 * 1024), 1),
                    "driver": driver,
                    "cuda": cuda_version,
                    "backend": "nvml",
                }
            )
    except Exception:
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return gpus


def _probe_gpus_smi() -> list[dict[str, Any]]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    gpus: list[dict[str, Any]] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            vram = float(parts[3])
        except ValueError:
            vram = None
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "vram_total_mb": vram,
                "driver": parts[4],
                "cuda": None,
                "backend": "nvidia-smi",
            }
        )
    return gpus


class GpuMonitor:
    """Samples GPU telemetry for a single device (default: index 0)."""

    def __init__(self, device_index: int = 0, enabled: bool = True) -> None:
        self.device_index = device_index
        self.backend: str = "none"
        self.available = False
        self._handle: Any = None
        self._pynvml: Any = None
        self._vram_total_mb: float | None = None
        if not enabled:
            self.backend = "disabled"
            return
        self._init()

    def _init(self) -> None:
        try:
            import pynvml  # type: ignore
        except Exception:
            self.backend = "nvidia-smi" if self._smi_available() else "none"
            self.available = self.backend == "nvidia-smi"
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._vram_total_mb = round(mem.total / (1024 * 1024), 1)
            self._pynvml = pynvml
            self.backend = "nvml"
            self.available = True
        except Exception:
            self.backend = "none"
            self.available = False

    def _smi_available(self) -> bool:
        try:
            out = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5, check=False
            )
            return out.returncode == 0 and bool(out.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            return False

    def sample(self, timestamp_ns: int) -> GpuSample | None:
        """Take one sample; returns ``None`` when GPU telemetry is unavailable."""
        if not self.available:
            return None
        if self.backend == "nvml":
            return self._sample_nvml(timestamp_ns)
        return self._sample_smi(timestamp_ns)

    def _sample_nvml(self, timestamp_ns: int) -> GpuSample | None:
        pynvml = self._pynvml
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            power_w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            temp = pynvml.nvmlDeviceGetTemperature(self._handle, 0)
            clock = pynvml.nvmlDeviceGetClockInfo(self._handle, 1)  # SM clock
            mem_clock = pynvml.nvmlDeviceGetClockInfo(self._handle, 2)  # memory clock
            try:
                pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(self._handle, 0) / 1024.0
                pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(self._handle, 1) / 1024.0
            except Exception:
                pcie_rx = pcie_tx = None
            return GpuSample(
                timestamp_ns=timestamp_ns,
                util_gpu_pct=float(util.gpu),
                util_mem_pct=float(util.memory),
                power_w=float(power_w),
                temperature_c=float(temp),
                vram_used_mb=round(mem.used / (1024 * 1024), 1),
                vram_total_mb=round(mem.total / (1024 * 1024), 1),
                clock_mhz=float(clock),
                mem_clock_mhz=float(mem_clock),
                pcie_rx_mb_s=pcie_rx,
                pcie_tx_mb_s=pcie_tx,
            )
        except Exception:
            return None

    def _sample_smi(self, timestamp_ns: int) -> GpuSample | None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device_index}",
                    "--query-gpu=utilization.gpu,utilization.memory,power.draw,"
                    "temperature.gpu,memory.used,memory.total,clocks.sm,clocks.mem",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        parts = [p.strip() for p in out.stdout.strip().split(",")]

        def num(value: str) -> float | None:
            try:
                return float(value)
            except ValueError:
                return None

        if len(parts) < 8:
            return None
        return GpuSample(
            timestamp_ns=timestamp_ns,
            util_gpu_pct=num(parts[0]),
            util_mem_pct=num(parts[1]),
            power_w=num(parts[2]),
            temperature_c=num(parts[3]),
            vram_used_mb=num(parts[4]),
            vram_total_mb=num(parts[5]),
            clock_mhz=num(parts[6]),
            mem_clock_mhz=num(parts[7]),
        )

    def close(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            self._pynvml = None
        self.available = False

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "available": self.available,
            "device_index": self.device_index,
            "vram_total_mb": self._vram_total_mb,
        }
