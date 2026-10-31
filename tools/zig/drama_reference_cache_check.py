"""GPU check for the Drama 3 reference cache: PCM sha256 with TF_DRAMA_REFERENCE_CACHE=0 against the default."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

TEXTS = ("The ferry leaves at seven, so we should be at the pier by half past six.",
         "A quiet library opens early on Saturdays and closes before dinner.",
         "Rain is expected tonight, but the market will still open in the morning.",
         "She folded the map, checked the compass, and started up the ridge.",
         "The bakery on the corner sells out of rye bread before noon.")
CACHE = {"off": "0", "on": None}


def plan(voice):
    """Each pass's requests in order: (step, voice, text); a list in a step is sent concurrently."""
    return [("fill", voice, TEXTS[0]), ("default", "default", TEXTS[1]), ("hit", voice, TEXTS[2]),
            ("repeat", voice, TEXTS[0]), [("lane-a", voice, TEXTS[3]), ("lane-b", voice, TEXTS[4])]]


def speak(port, voice, text, timeout):
    body = json.dumps({"model": "fishaudio/drama-3", "input": text, "voice": voice,
                       "response_format": "pcm", "stream": True}).encode()
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


def ready(port, process, deadline):
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with status {process.returncode} before readiness")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    raise TimeoutError("server not ready")


def run_pass(args, rows, cache, log):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**os.environ, "TF_DRAMA_PREFILL_ROWS": str(rows)}
    env.pop("TF_DRAMA_REFERENCE_CACHE", None)
    if CACHE[cache] is not None:
        env["TF_DRAMA_REFERENCE_CACHE"] = CACHE[cache]
    command = [str(args.bin), "serve", str(args.checkpoint), "--host", "127.0.0.1", "--port", str(port),
               "--name", "fishaudio/drama-3", "--context", "8192", "--parallel", "2", "--no-drafts",
               "--prompt-cache-gib", "0", "--keep-warm", "0", "--voices", str(args.voices)]
    with log.open("w") as output:
        server = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            ready(port, server, time.monotonic() + args.ready_timeout)
            replies = {}
            for step in plan(args.voice):
                group = step if isinstance(step, list) else [step]
                with ThreadPoolExecutor(len(group)) as pool:
                    futures = {name: pool.submit(speak, port, voice, text, args.timeout) for name, voice, text in group}
                replies.update({name: future.result() for name, future in futures.items()})
        finally:
            server.terminate()
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    text = log.read_text(errors="replace")
    return {"replies": replies, "cache_hits": text.count("reference cache hit"),
            "cache_builds": text.count("reference cache built")}


def verdict(passes):
    """Every reply's PCM equal with the cache on and off at each prefill size; the repeat equals its fill."""
    problems = []
    for rows, by_cache in passes.items():
        off, on = by_cache["off"], by_cache["on"]
        for name, reply in off["replies"].items():
            if on["replies"][name]["sha256"] != reply["sha256"]:
                problems.append(f"rows={rows} {name}: PCM differs with the cache on")
        for cache, result in by_cache.items():
            if result["replies"]["repeat"]["sha256"] != result["replies"]["fill"]["sha256"]:
                problems.append(f"rows={rows} cache={cache}: the repeated request's PCM differs from the first")
        if off["cache_hits"] != 0:
            problems.append(f"rows={rows}: the cache hit with TF_DRAMA_REFERENCE_CACHE=0")
        if on["cache_hits"] == 0:
            problems.append(f"rows={rows}: no reference cache hit with the cache on")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, required=True, help="tensorfold-native built with this change (CUDA)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="fishaudio/drama-3 checkpoint directory")
    parser.add_argument("--voices", type=Path, required=True, help="voice catalog directory for --voices")
    parser.add_argument("--voice", required=True, help="a catalog voice in --voices")
    parser.add_argument("--rows", default="1,128", help="TF_DRAMA_PREFILL_ROWS values, comma separated")
    parser.add_argument("--out", type=Path, required=True, help="new directory for server logs and results.json")
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    passes = {}
    for rows in (int(r) for r in args.rows.split(",")):
        passes[rows] = {cache: run_pass(args, rows, cache, args.out / f"server-rows{rows}-cache-{cache}.log")
                        for cache in CACHE}
    problems = verdict(passes)
    result = {"pass": not problems, "problems": problems, "passes": passes,
              "hit_first_audio_s": {rows: {cache: by_cache[cache]["replies"]["hit"]["first_audio_s"]
                                           for cache in CACHE} for rows, by_cache in passes.items()}}
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": result["pass"], "problems": problems, "hit_first_audio_s": result["hit_first_audio_s"]},
                     indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
