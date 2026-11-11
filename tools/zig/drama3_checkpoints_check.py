"""Drama 3's FP8 and pruned NVFP4 checkpoints served from one build: experts held, fixed-seed replies, timings."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.request

from drama_guidance_check import require_reference
from drama_reference_cache_check import TEXTS, ready

# 10,240 routed experts in all; FP8 holds every one, the published compact checkpoints their kept experts
HELD = {"fp8": 10240, "pruned": 7369, "pruned-5900": 5900}
HELD_LINE = re.compile(r"routed experts held (\d+) of 10240")
SEED = 7
PCM_BYTES_PER_S = 44100 * 2
MODEL = "fishaudio/drama-3"
# each solo reply once more, then both at once on the server's two lanes
SOLO = [("whole", TEXTS[0], False), ("stream", TEXTS[1], True)]


def speak(port, text, stream, timeout):
    """One fixed-seed reply as PCM: sha256, bytes, seconds to its first chunk and to its end."""
    body = json.dumps({"model": MODEL, "input": text, "response_format": "pcm", "stream": stream, "seed": SEED}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/audio/speech", body,
                                     {"Content-Type": "application/json"})
    digest, size, first = hashlib.sha256(), 0, None
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        while chunk := response.read(65536):
            first = first if first is not None else time.monotonic() - started
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size, "first_audio_s": first, "total_s": time.monotonic() - started}


def peak_bytes(memory):
    """The device peak where the backend counts one (CUDA), else the process's lifetime peak footprint (macOS)."""
    if "peak_bytes" in memory.get("device", {}):
        return memory["device"]["peak_bytes"], "device.peak_bytes"
    return memory["process"]["lifetime_peak_physical_footprint_bytes"], "process.lifetime_peak_physical_footprint_bytes"


def run_checkpoint(args, checkpoint, log):
    """Serve one checkpoint, send the solo replies twice and then together; its log, replies, load time and peak."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [str(args.bin), "serve", str(checkpoint), "--host", "127.0.0.1", "--port", str(port), "--name", MODEL,
               "--parallel", "2", "--voices", str(args.voices), "--default-voice", args.voice,
               "--cfg-gamma", "1.5", "--cfg-batched"]
    with log.open("w") as output:
        started = time.monotonic()
        server = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        try:
            ready(port, server, started + args.ready_timeout)
            load_s = time.monotonic() - started
            replies = {}
            for attempt in ("first", "repeat"):
                for mode, text, stream in SOLO:
                    replies[f"{mode}-{attempt}"] = speak(port, text, stream, args.timeout)
            with ThreadPoolExecutor(len(SOLO)) as pool:
                lanes = {f"{mode}-lane": pool.submit(speak, port, text, stream, args.timeout)
                         for mode, text, stream in SOLO}
            replies.update({key: future.result() for key, future in lanes.items()})
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/memory", timeout=10) as response:
                peak, source = peak_bytes(json.load(response))
        finally:
            server.terminate()
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    held = HELD_LINE.findall(log.read_text(errors="replace"))
    stream = replies["stream-first"]
    audio_s = stream["bytes"] / PCM_BYTES_PER_S
    return {"held": int(held[-1]) if held else None, "load_s": load_s, "peak_bytes": peak, "peak_source": source,
            "first_audio_s": stream["first_audio_s"], "total_s": stream["total_s"], "audio_s": audio_s,
            "rtf": stream["total_s"] / audio_s if audio_s else None, "replies": replies}


def verdict(runs):
    """Every checkpoint holds its experts and speaks; each reply equals its repeat and its concurrent lane."""
    problems = []
    for name, run in runs.items():
        if run["held"] != HELD[name]:
            problems.append(f"{name}: routed experts held {run['held']}, expected {HELD[name]}")
        replies = run["replies"]
        problems += [f"{name} {key}: no PCM" for key, reply in replies.items() if reply["bytes"] == 0]
        for mode, _, _ in SOLO:
            first = replies[f"{mode}-first"]["sha256"]
            for other in ("repeat", "lane"):
                if replies[f"{mode}-{other}"]["sha256"] != first:
                    problems.append(f"{name} {mode}: the {other} reply's PCM differs from the first")
    return problems


def table(runs):
    """The receipt's Markdown table, one row per checkpoint."""
    rows = ["| Checkpoint | Experts held | Load (s) | Peak memory (GiB) | First audio (s) | Total (s) | Audio (s) | RTF |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in runs.items():
        rows.append(f"| {name} | {r['held']} | {r['load_s']:.1f} | {r['peak_bytes'] / 2**30:.2f} | "
                    f"{r['first_audio_s']:.2f} | {r['total_s']:.2f} | {r['audio_s']:.2f} | {r['rtf']:.3f} |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, required=True, help="tensorfold-native built from main")
    for name in HELD:
        parser.add_argument(f"--{name}", type=Path, help=f"the {name} checkpoint directory; left out, not served")
    parser.add_argument("--voices", type=Path, required=True, help="voice catalog directory for --voices")
    parser.add_argument("--voice", required=True, help="a catalog voice in --voices with a reference")
    parser.add_argument("--out", type=Path, required=True, help="new directory for server logs and results.json")
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    require_reference(args.voices, args.voice)
    checkpoints = {name: path for name in HELD if (path := getattr(args, name.replace("-", "_")))}
    if not checkpoints:
        parser.error("name at least one checkpoint")
    args.out.mkdir(parents=True, exist_ok=False)
    runs = {name: run_checkpoint(args, path, args.out / f"server-{name}.log") for name, path in checkpoints.items()}
    problems = verdict(runs)
    (args.out / "results.json").write_text(json.dumps({"pass": not problems, "problems": problems, "runs": runs},
                                                      indent=2) + "\n")
    print(table(runs))
    print(json.dumps({"pass": not problems, "problems": problems}, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
