"""GPU check for Drama 3 semantic CFG on CUDA: paired and separate guides give the same PCM, and gamma 1 is plain."""
import argparse
import json
import os
from pathlib import Path
import sys

from drama_reference_cache_check import TEXTS, serve

GAMMAS = ("1", "1.5", "3")
CFG = ["--cfg-gamma", "1.5", "--cfg-allow-request-gamma", "--cfg-force-guided"]
MODES = {"plain": [], "separate": CFG, "paired": [*CFG, "--cfg-batched"]}
# One row a chunk, and chunks long enough that a cached reference's suffix (a short text) is one round; a reference
# keeps whole chunks only, so a voice needs more reference rows than a chunk for its later requests to hit the cache.
ROWS = (1, 64)


def plan(voice):
    """(step, voice, text, headers) per fixed text and gamma, then two guided lanes at once."""
    steps = [(f"t{t}-g{g}", voice, TEXTS[t], {"X-Cfg-Gamma": g}) for t in range(3) for g in GAMMAS]
    return [*steps, [("lane-a", voice, TEXTS[3], {"X-Cfg-Gamma": "3"}),
                     ("lane-b", voice, TEXTS[4], {"X-Cfg-Gamma": "1.5"})]]


def require_reference(voices, voice):
    """Guidance runs only for a reference-backed catalog voice; the built-in default has no reference."""
    path = voices / f"{voice}.json"
    if voice.lower() == "default" or not path.is_file() or not json.loads(path.read_text()).get("references"):
        raise SystemExit(f"--voice {voice!r} must name a catalog voice with a reference in {voices}")


def verdict(runs):
    """Every reply has PCM; paired equals separate; forced gamma 1 equals the server without CFG and gamma above 1
    differs from it; every guided server resumed a cached reference; per prefill size."""
    problems = []
    for rows, modes in runs.items():
        separate, paired, plain = modes["separate"], modes["paired"], modes["plain"]
        for mode, run in modes.items():
            problems += [f"rows={rows} {name}: {mode} returned no PCM" for name, reply in run["replies"].items()
                         if reply["bytes"] == 0]
        for name, reply in separate["replies"].items():
            if paired["replies"][name]["sha256"] != reply["sha256"]:
                problems.append(f"rows={rows} {name}: paired PCM differs from separate")
        for name, reply in plain["replies"].items():
            for mode in ("separate", "paired"):
                guided = modes[mode]["replies"]
                if guided[name]["sha256"] != reply["sha256"]:
                    problems.append(f"rows={rows} {name}: {mode} gamma 1 differs from the server without CFG")
                for gamma in GAMMAS[1:]:
                    mixed = name.removesuffix("-g1") + f"-g{gamma}"
                    if guided[mixed]["sha256"] == guided[name]["sha256"]:
                        problems.append(f"rows={rows} {mixed}: {mode} PCM equals gamma 1, so guidance changed nothing")
        for mode in ("separate", "paired"):
            if modes[mode]["cache_hits"] == 0:
                problems.append(f"rows={rows}: {mode} never resumed a cached reference")
    return problems


def run_mode(args, env, flags, log, steps):
    replies = serve(args, env, flags, log, steps)
    return {"replies": replies, "cache_hits": log.read_text(errors="replace").count("reference cache hit")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, required=True, help="tensorfold-native built with this change (CUDA)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="fishaudio/drama-3 checkpoint directory")
    parser.add_argument("--voices", type=Path, required=True, help="voice catalog directory for --voices")
    parser.add_argument("--voice", required=True, help="a catalog voice in --voices whose reference has over 64 rows")
    parser.add_argument("--out", type=Path, required=True, help="new directory for server logs and results.json")
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    require_reference(args.voices, args.voice)
    args.out.mkdir(parents=True, exist_ok=False)
    steps = plan(args.voice)
    plain = [s for s in steps if not isinstance(s, list) and s[0].endswith("-g1")]
    runs = {}
    for rows in ROWS:
        # each voice's later requests resume after its cached reference: a one-chunk suffix at large rows
        env = {**os.environ, "TF_DRAMA_PREFILL_ROWS": str(rows), "TF_DRAMA_REFERENCE_CACHE": "2"}
        runs[rows] = {mode: run_mode(args, env, flags, args.out / f"server-rows{rows}-{mode}.log",
                                     plain if mode == "plain" else steps) for mode, flags in MODES.items()}
    problems = verdict(runs)
    result = {"pass": not problems, "problems": problems, "runs": runs}
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": result["pass"], "problems": problems}, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
