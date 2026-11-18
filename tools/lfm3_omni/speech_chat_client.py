"""Streams spoken turns through POST /v1/audio/chat and times them: speech_chat_client.py HOST:PORT OUT_DIR WAV..."""
import argparse
import base64
import http.client
import json
import os
import statistics
import subprocess
import time
import wave
from pathlib import Path


def busy():
    """True while $LFM3_OMNI_BUSY (a command) exits 0."""
    cmd = os.environ.get("LFM3_OMNI_BUSY")
    return bool(cmd) and subprocess.run(cmd, shell=True).returncode == 0


def turn(host, port, body):
    """One request read as it arrives: the reply's text, its 16-bit 24 kHz PCM, client-side times and the done event."""
    c = http.client.HTTPConnection(host, port, timeout=600)
    t0 = time.perf_counter()
    c.request("POST", "/v1/audio/chat", body=json.dumps(body), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {r.read()[:200]!r}")
    first_text = first_audio = None
    pcm, text, done = bytearray(), "", {}
    while line := r.readline():
        line = line.strip()
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        ev = json.loads(line[6:])
        if ev["type"] == "text":
            first_text = first_text or time.perf_counter()
            text += ev["delta"]
        elif ev["type"] == "audio":
            first_audio = first_audio or time.perf_counter()
            pcm += base64.b64decode(ev["data"])
        elif ev["type"] == "done":
            done = ev
    total = time.perf_counter() - t0
    return text.strip(), bytes(pcm), {
        "first_text_s": round((first_text or t0) - t0, 3), "first_audio_s": round((first_audio or t0) - t0, 3),
        "total_s": round(total, 3), "speech_s": round(len(pcm) / 48000, 2),
        "server_first_audio_ms": done.get("first_audio_ms"), "frames": done.get("frames")}


def main():
    """Every WAV in order (seed + its index), each reply saved as OUT_DIR/rNN.wav, medians in OUT_DIR/replies.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("server")
    ap.add_argument("out")
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--system-file")
    ap.add_argument("--audio-temperature", type=float, default=0.4)
    ap.add_argument("--audio-repetition-penalty", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    host, port = a.server.rsplit(":", 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    system = Path(a.system_file).read_text().strip() if a.system_file else None
    rows = []
    for i, w in enumerate(a.wavs):
        while busy():
            time.sleep(5)
        body = {"audio": base64.b64encode(Path(w).read_bytes()).decode(), "stream": True, "seed": a.seed + i,
                "audio_temperature": a.audio_temperature, "audio_repetition_penalty": a.audio_repetition_penalty}
        if system is not None:
            body["system"] = system
        text, pcm, row = turn(host, int(port), body)
        with wave.open(str(out / f"r{i:02d}.wav"), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(pcm)
        rows.append({"wav": Path(w).name, "reply": text} | row)
        print(json.dumps(rows[-1]), flush=True)
    keys = ("first_text_s", "first_audio_s", "total_s", "speech_s")
    median = {k: round(statistics.median(r[k] for r in rows), 3) for k in keys}
    print("MEDIAN", json.dumps(median))
    json.dump({"system": system, "median": median, "rows": rows}, open(out / "replies.json", "w"), indent=1)


if __name__ == "__main__":
    main()
