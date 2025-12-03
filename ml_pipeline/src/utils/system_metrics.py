import os
import time

try:
    import psutil
except ImportError as exc:
    raise ImportError(
        "psutil is required to gather CPU and memory usage metrics. Install it with 'pip install psutil'."
    ) from exc


class SystemMetricsTracker:
    """Track wall-clock time, CPU usage, and peak RSS for the current process."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_wall = time.perf_counter()
        self.start_cpu = self.process.cpu_times()
        self.peak_rss = self.process.memory_info().rss

    def sample_memory(self):
        rss = self.process.memory_info().rss
        if rss > self.peak_rss:
            self.peak_rss = rss
        return rss

    def finish(self):
        end_wall = time.perf_counter()
        end_cpu = self.process.cpu_times()
        self.sample_memory()

        wall_time = end_wall - self.start_wall
        cpu_time = (end_cpu.user - self.start_cpu.user) + (end_cpu.system - self.start_cpu.system)
        avg_cpu_percent = (cpu_time / wall_time) * 100 if wall_time > 0 else 0.0

        return {
            'wall_time_sec': wall_time,
            'cpu_time_sec': cpu_time,
            'avg_cpu_percent': avg_cpu_percent,
            'peak_rss_bytes': self.peak_rss,
        }
