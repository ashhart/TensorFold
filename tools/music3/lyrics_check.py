"""Lyric accuracy of music3-generate: songs through --serve, then Whisper (faster-whisper large-v3-turbo on the CPU,
the Studio's own qa.transcribe) against the lyrics with the Studio's music.heard score.

    python tools/music3/lyrics_check.py --bin music3-generate --studio-app APP --prompts DIR --out DIR
        [--weights int8 bf16] [--seconds 45] [--songs hiphop:7 pop:11 long:5]

The engine runs first and ends before Whisper loads (no two models at once). Writes DIR/lyrics.json.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def serve_songs(binary: str, weights: str, songs: list, prompts: Path, out: Path, seconds: float) -> list:
    proc = subprocess.Popen([binary, "--serve", "1", "--weights", weights], stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, text=True, bufsize=1)
    answers = []

    def read() -> dict:
        for line in proc.stderr:
            if line.startswith("{"):
                msg = json.loads(line)
                if "progress" not in msg:
                    return msg
        raise RuntimeError("the engine stopped")

    ready = read()
    assert ready.get("ready"), ready
    for name, seed in songs:
        wav = out / f"{name}_{seed}_{weights}.wav"
        req = {"id": f"{name}-{seed}-{weights}", "caption": (prompts / f"{name}.caption").read_text(),
               "lyrics": (prompts / f"{name}.lyrics").read_text(), "wav": str(wav), "seconds": seconds, "seed": seed}
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        ans = read()
        ans["name"], ans["weights"], ans["seed"] = name, weights, seed
        answers.append(ans)
        print(json.dumps(ans), flush=True)
    proc.stdin.close()
    proc.wait(timeout=60)
    return answers


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bin", required=True)
    p.add_argument("--studio-app", required=True, type=Path)
    p.add_argument("--prompts", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--weights", nargs="+", default=["int8", "bf16"])
    p.add_argument("--seconds", type=float, default=45)
    p.add_argument("--songs", nargs="+", default=["hiphop:7", "pop:11", "long:5"])
    p.add_argument("--extra", nargs="*", default=[], help="NAME=WAV:LYRICS more songs to score (a ComfyUI song)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    songs = [(s.split(":")[0], int(s.split(":")[1])) for s in a.songs]
    made = []
    for w in a.weights:
        made += serve_songs(a.bin, w, songs, a.prompts, a.out, a.seconds)
    sys.path.insert(0, str(a.studio_app))
    import music  # noqa: E402
    import qa  # noqa: E402
    rows = []
    todo = [(m["name"], m["weights"], m["seed"], Path(m["wav"]), a.prompts / f"{m['name']}.lyrics") for m in made if m.get("ok")]
    for e in a.extra:
        label, rest = e.split("=", 1)
        wav, lyr = rest.split(":", 1)
        todo.append((label, "comfyui", 0, Path(wav), Path(lyr)))
    for name, weights, seed, wav, lyr in todo:
        t0 = time.time()
        heard = qa.transcribe(wav)
        score = music.heard(lyr.read_text(), heard or "")
        rows.append({"song": name, "seed": seed, "weights": weights, "wav": str(wav), "score": score["score"], "found": score["found"],
                     "words": score["words"], "reached": score["reached"], "total": score["total"], "whisper_s": round(time.time() - t0, 1),
                     "heard": heard})
        print(json.dumps(rows[-1]), flush=True)
    (a.out / "lyrics.json").write_text(json.dumps({"songs": made, "scores": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
