"""Drive music3-generate --serve 1 through the Studio's own client (app/music.py's Mlx route, voice.Served).

Copy this file beside a Studio checkout's app/ folder (main 86ae680 or later) and run it with the Studio venv:
    python served_test.py PROMPTS OUT_DIR       (env MUSIC3_BIN, default music3-generate on PATH)
"""
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "app"))
os.environ.setdefault("MUSIC3_GUARD_GB", "25")
import music  # noqa: E402

BIN = os.path.expanduser(os.environ.get("MUSIC3_BIN", "music3-generate"))


class Cuda(music.Mlx):
    name, label = "cuda", "TensorFold CUDA · DGX Spark"
    def argv(self):
        return [BIN, "--serve", "1"]
    def check(self):
        return ""


def main():
    P = Path(sys.argv[1]).expanduser()  # hiphop.caption, hiphop.lyrics, pop.caption, pop.lyrics
    out = Path(sys.argv[2]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    s = Cuda()
    seen = []
    for i, (name, secs, seed) in enumerate((("hiphop", 20, 7), ("pop", 15, 11))):
        job = {"id": f"served-{i}", "status": "running", "progress": None}
        params = {"caption": (P / f"{name}.caption").read_text(), "lyrics": (P / f"{name}.lyrics").read_text(), "seconds": secs,
                  "seed": seed, "ar_cfg": 1.5, "top_k": 50, "steps": 30, "dit_cfg": 1.7}
        t0 = time.time()
        res = s.make(job, params, out / f"{name}.wav")
        seen.append({"request": i, "result": res, "ready": s.ready, "wall_s": round(time.time() - t0, 1),
                     "last_progress": job.get("progress"), "stats_file": json.loads((out / f"{name}.stats.json").read_text()) if (out / f"{name}.stats.json").exists() else None})
        print(json.dumps(seen[-1]), flush=True)
    s.stop("test done")
    print("events", json.dumps(s.events))
    print("log tail", s.log[-6:])


if __name__ == "__main__":
    main()
