import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import pynvml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pynvml = None


@dataclass
class GpuSample:
    timestamp: float
    gpu_util: float
    mem_util: float
    mem_used_mb: float
    temperature: float
    power_watts: float


@dataclass
class GpuStats:
    samples: List[GpuSample] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        if not self.samples:
            return {}
        gpu_utils = [s.gpu_util for s in self.samples]
        mem_utils = [s.mem_util for s in self.samples]
        mem_used = [s.mem_used_mb for s in self.samples]
        temps = [s.temperature for s in self.samples]
        power = [s.power_watts for s in self.samples]
        return {
            "gpu_util_avg": float(sum(gpu_utils) / len(gpu_utils)),
            "gpu_util_max": float(max(gpu_utils)),
            "mem_util_avg": float(sum(mem_utils) / len(mem_utils)),
            "mem_util_max": float(max(mem_utils)),
            "mem_used_avg_mb": float(sum(mem_used) / len(mem_used)),
            "mem_used_max_mb": float(max(mem_used)),
            "temp_max_c": float(max(temps)),
            "power_avg_w": float(sum(power) / len(power)),
            "power_max_w": float(max(power)),
        }


class GpuMonitor:
    def __init__(self, device_index: int = 0, interval: float = 0.5):
        self.device_index = device_index
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.stats = GpuStats()
        self._nvml_initialized = False
        self.disabled_reason: Optional[str] = None

    def _init_nvml(self):
        if pynvml is None:
            self._nvml_initialized = False
            self.disabled_reason = "pynvml is not installed"
            return
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
        except pynvml.NVMLError as exc:
            self._nvml_initialized = False
            self.disabled_reason = f"NVML init failed: {exc}"

    def _collect(self):
        if not self._nvml_initialized:
            return

        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        while not self._stop_event.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            self.stats.samples.append(
                GpuSample(
                    timestamp=time.time(),
                    gpu_util=float(util.gpu),
                    mem_util=float(util.memory),
                    mem_used_mb=float(mem.used) / (1024 * 1024),
                    temperature=float(temp),
                    power_watts=float(power),
                )
            )
            time.sleep(self.interval)

    def start(self):
        self._init_nvml()
        if not self._nvml_initialized:
            # Best-effort: skip monitoring when NVML is unavailable.
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
            finally:
                self._nvml_initialized = False

    def summary(self) -> Dict[str, float]:
        return self.stats.summary()
