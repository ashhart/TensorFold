"""Own one native server and compare raw/chat plain/drafted replies with the selected CLI receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
from urllib.request import Request, urlopen

from qwen27_gate import PROMPTS


def check(reply, ids, tokens, drafted):
    runtime, usage = reply["tensorfold"], reply["usage"]
    digest = hashlib.sha256(",".join(str(token) for token in tokens).encode()).hexdigest()[:12]
    if usage["prompt_tokens"] != len(ids) or usage["completion_tokens"] != len(tokens):
        raise ValueError("served prompt/generation lengths differ from CLI")
    if runtime["token_sha"] != digest or runtime["drafts"] is not drafted:
        raise ValueError("served token SHA or draft mode differs from CLI")
    speculative = reply["speculative"]
    if drafted and (speculative["rounds"] <= 0 or speculative["drafted"] <= 0):
        raise ValueError("drafted server request did not exercise draft verification")
    return {"token_sha": digest, "drafts": drafted, "usage": usage,
            "runtime": runtime, "speculative": speculative}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--drafter", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18436)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError("invalid private server port")
    prompts = json.loads(args.prompts.read_text())
    cli = json.loads(args.cli.read_text())
    if set(prompts) != set(PROMPTS) or set(cli) != set(PROMPTS):
        raise ValueError("expected all four fixed public prompts")
    for name in PROMPTS:
        native = cli[name]["native"]
        if len(native["tokens"]) != 256 or not native["drafted_equal_plain"] or not native["final_state_equal_plain"]:
            raise ValueError("selected native draft/plain CLI contract missing")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    args.out.mkdir(parents=True, exist_ok=False)
    command = [str(args.binary), "serve", str(args.model), "--drafter", str(args.drafter),
               "--drafter-bits", "4", "--context", "4096", "--name", "qwen27",
               "--host", "127.0.0.1", "--port", str(args.port), "--no-thinking"]
    base = f"http://127.0.0.1:{args.port}"
    receipts = {}
    with (args.out / "server.log").open("x") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        print(f"START native HTTP server owned PID{child.pid}", flush=True)
        try:
            deadline = time.monotonic() + 180
            while True:
                if child.poll() is not None:
                    raise RuntimeError("owned server exited during startup")
                try:
                    with urlopen(base + "/health", timeout=2) as response:
                        json.load(response)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("owned server startup deadline") from None
                    time.sleep(0.2)
            for name, text in PROMPTS.items():
                for chat in (False, True):
                    for drafted in (False, True):
                        body = {"model": "qwen27", "max_tokens": 256, "temperature": 0,
                                "draft": drafted, "ignore_eos": True}
                        body.update({"messages": [{"role": "user", "content": text}],
                                     "chat_template_kwargs": {"enable_thinking": False}} if chat
                                    else {"prompt": prompts[name]})
                        endpoint = "/v1/chat/completions" if chat else "/v1/completions"
                        request = Request(base + endpoint, data=json.dumps(body).encode(),
                                          headers={"Content-Type": "application/json"})
                        with urlopen(request, timeout=180) as response:
                            reply = json.load(response)
                        key = f"{name}-{'chat' if chat else 'raw'}-{'draft' if drafted else 'plain'}"
                        with (args.out / f"{key}.json").open("x") as output:
                            output.write(json.dumps(reply, indent=2) + "\n")
                        receipts[key] = check(reply, prompts[name], cli[name]["native"]["tokens"], drafted)
                        print(f"PASS {key} 256 tokens and rendered prompt length", flush=True)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
            print(f"END native HTTP server owned PID{child.pid} rc{child.returncode}", flush=True)
    with (args.out / "gate.json").open("x") as output:
        output.write(json.dumps({"cases": receipts, "passed": len(receipts)}, indent=2) + "\n")
    if len(receipts) != 16:
        raise RuntimeError("incomplete HTTP gate")


if __name__ == "__main__":
    main()
