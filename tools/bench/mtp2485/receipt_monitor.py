"""GPU process discovery and memory sampling for the local receipt runners."""
from __future__ import annotations
import json
import subprocess
import threading
import time
from pathlib import Path


def nvidia(*args: str) -> str:
    p = subprocess.run(['nvidia-smi', *args], text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=10)
    if p.returncode:
        raise RuntimeError(f'nvidia-smi failed: {p.stdout.strip()}')
    return p.stdout.strip()


def compute_apps() -> list[dict]:
    raw = nvidia('--query-compute-apps=pid,process_name,used_memory',
                 '--format=csv,noheader,nounits')
    apps = []
    for line in raw.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) < 3:
            continue
        try:
            used = int(fields[2])
        except ValueError:
            used = None
        apps.append({'pid': fields[0], 'process': fields[1], 'used_mib': used})
    return apps


class Monitor:
    def __init__(self, pid: int, path: Path):
        self.pid, self.path = pid, path
        self.stop = threading.Event()
        self.rows: list[dict] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            now = time.time()
            try:
                raw = nvidia('--query-gpu=memory.used,memory.total,utilization.gpu',
                             '--format=csv,noheader,nounits')
                gpu_fields = [x.strip() for x in raw.splitlines()[0].split(',')]
                apps = compute_apps()
                proc_gpu = max((a['used_mib'] for a in apps if a['pid'] == str(self.pid) and a['used_mib'] is not None), default=None)
                rss = None
                status = Path(f'/proc/{self.pid}/status')
                if status.exists():
                    for line in status.read_text().splitlines():
                        if line.startswith('VmRSS:'):
                            rss = int(line.split()[1]) // 1024
                            break
                row = {'time_unix': now,
                       'gpu_used_mib': int(gpu_fields[0]),
                       'gpu_total_mib': int(gpu_fields[1]),
                       'gpu_util_pct': int(gpu_fields[2]),
                       'process_gpu_mib': proc_gpu,
                       'process_rss_mib': rss}
                self.rows.append(row)
                with self.path.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row) + '\n')
            except Exception as e:  # Retain monitor faults without interrupting the served test.
                self.rows.append({'time_unix': now, 'monitor_error': str(e)})
            self.stop.wait(1.0)

    def finish(self) -> dict:
        self.stop.set()
        self.thread.join(timeout=15)
        good = [r for r in self.rows if 'gpu_used_mib' in r]
        return {
            'samples': len(good),
            'gpu_peak_used_mib': max((r['gpu_used_mib'] for r in good), default=None),
            'gpu_peak_process_mib': max((r['process_gpu_mib'] for r in good if r['process_gpu_mib'] is not None), default=None),
            'process_peak_rss_mib': max((r['process_rss_mib'] for r in good if r['process_rss_mib'] is not None), default=None),
            'monitor_errors': [r['monitor_error'] for r in self.rows if 'monitor_error' in r],
        }
