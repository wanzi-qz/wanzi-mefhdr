"""简单计时器：统计各阶段耗时（秒），统一入 JSON。"""
import json
import time
from pathlib import Path


class Timer:
    def __init__(self):
        self.records = {}
        self._t0 = time.perf_counter()

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def stage(self, name):
        """上下文管理器用法：with timer.stage('align'): ..."""
        return _Stage(self, name)

    def _add(self, name, sec):
        self.records[name] = round(self.records.get(name, 0.0) + sec, 4)

    def now(self):
        return time.perf_counter() - self._t0


class _Stage:
    def __init__(self, timer, name):
        self.timer, self.name = timer, name

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.timer._add(self.name, time.perf_counter() - self.t)


def save_timing(timer: Timer, out_dir, extra=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"timing_sec": timer.records, "total_sec": round(timer.now(), 4)}
    if extra:
        payload.update(extra)
    with open(out / "timing.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload
