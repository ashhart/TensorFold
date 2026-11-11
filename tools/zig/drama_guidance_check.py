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


def plan(voice):
    """(step, voice, text, headers) per fixed text and gamma, then two guided lanes at once."""
    steps = [(f"t{t}-g{g}", voice, TEXTS[t], {"X-Cfg-Gamma": g}) for t in range(3) for g in GAMMAS]
    return [*steps, [("lane-a", voice, TEXTS[3], {"X-Cfg-Gamma": "3"}),
                     ("lane-b", voice, TEXTS[4], {"X-Cfg-Gamma": "1.5"})]]


def verdict(modes):
    """Paired equals separate on every reply; forced gamma 1 equals the server without CFG."""
    problems = []
    separate, paired, plain = modes["separate"], modes["paired"], modes["plain"]
    for name, reply in separate.items():
        if paired[name]["sha256"] != reply["sha256"]:
            problems.append(f"{name}: paired PCM differs from separate")
    for name, reply in plain.items():
        if name.endswith("-g1"):
            for mode in ("separate", "paired"):
                if modes[mode][name]["sha256"] != reply["sha256"]:
                    problems.append(f"{name}: {mode} gamma 1 differs from the server without CFG")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, required=True, help="tensorfold-native built with this change (CUDA)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="fishaudio/drama-3 checkpoint directory")
    parser.add_argument("--voices", type=Path, required=True, help="voice catalog directory for --voices")
    parser.add_argument("--voice", required=True, help="a catalog voice in --voices")
    parser.add_argument("--out", type=Path, required=True, help="new directory for server logs and results.json")
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    steps = plan(args.voice)
    plain = [s for s in steps if not isinstance(s, list) and s[0].endswith("-g1")]
    modes = {mode: serve(args, dict(os.environ), flags, args.out / f"server-{mode}.log",
                         plain if mode == "plain" else steps) for mode, flags in MODES.items()}
    problems = verdict(modes)
    result = {"pass": not problems, "problems": problems, "modes": modes}
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": result["pass"], "problems": problems}, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
