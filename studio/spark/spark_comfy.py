"""The studio's ComfyUI server on a DGX Spark: start and free it, upload inputs, run a workflow and fetch its files."""
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import memory_guard  # beside this file

HERE = Path(__file__).resolve().parent
PREFIX = Path(os.environ.get("PREFIX", Path.home() / ".local/opt/tensorfold-studio"))
URL = os.environ.get("SPARK_COMFY_URL", "http://127.0.0.1:" + os.environ.get("SPARK_COMFY_PORT", "8190"))  # the port spark/comfy.sh starts it on
STATE = PREFIX / "comfy.kind"
MODELS = Path(os.environ.get("SPARK_MODELS", Path.home() / "spark-studio-models"))
TEXT_TAG = ""
UPLOADED: list[str] = []  # what this run put in ComfyUI's input folder


def call(path, data=None):
    request = urllib.request.Request(URL + path, data=json.dumps(data).encode() if data is not None else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as reply:
        body = reply.read()
    return json.loads(body) if body else {}


def up() -> bool:
    try:
        call("/system_stats")
        return True
    except OSError:
        return False


def ensure_server(kind: str) -> None:
    """Start ComfyUI if it is not running; unload the other family's models when the job kind changes."""

    if not up():
        print("[tensorfold] starting ComfyUI (first render after a start loads the models)", flush=True)
        subprocess.run(["bash", str(HERE / "comfy.sh"), "start"], check=True)
        for _ in range(180):
            if up():
                break
            time.sleep(2)
        else:
            sys.exit(f"ComfyUI did not answer at {URL}: see {PREFIX}/comfy.log")
    kind += TEXT_TAG
    last = STATE.read_text().strip() if STATE.exists() else ""
    if last and last != kind:
        # the picture and video models together would not leave room for the render itself in 128 GB
        call("/free", {"unload_models": True, "free_memory": True})
    STATE.write_text(kind)


def fresh_server(why: str) -> None:
    """Stop the ComfyUI server and wait until its memory is back: the next render starts with nothing loaded."""

    print(f"[tensorfold] restarting ComfyUI: {why}", flush=True)
    subprocess.run(["bash", str(HERE / "comfy.sh"), "stop"], check=False)
    for _ in range(60):
        if not up():
            break
        time.sleep(1)
    before = -1
    for _ in range(90):  # the memory comes back a little after the process is gone
        free = memory_guard.meminfo().get("MemAvailable", 0)
        if free - before < 200:
            break
        before = free
        time.sleep(1)


def upload(image: str) -> str:
    """Put an image (or a video or a sound file) in ComfyUI's input folder; returns the name its loader wants."""

    path = Path(image)
    name = f"studio_{uuid.uuid4().hex[:10]}{path.suffix.lower() or '.png'}"
    UPLOADED.append(name)
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + (
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n").encode()
    request = urllib.request.Request(URL + "/upload/image", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(request, timeout=120) as reply:
        return json.loads(reply.read())["name"]


async def run(workflow: dict, sampler: str) -> str:
    """Queue a workflow and print the sampler's steps until it ends (socket opened first, so none is missed); its id."""

    import aiohttp

    client = uuid.uuid4().hex
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(URL.replace("http", "ws", 1) + f"/ws?clientId={client}", max_msg_size=0) as ws:
            prompt_id = call("/prompt", {"prompt": workflow, "client_id": client})["prompt_id"]
            stages, running, since = [], None, time.time()
            async for message in ws:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                event = json.loads(message.data)
                data = event.get("data", {})
                if data.get("prompt_id") not in (None, prompt_id):
                    continue
                if event["type"] == "progress" and str(data.get("node")) == sampler:
                    print(f"[tensorfold] step {data['value']}/{data['max']}", flush=True)
                elif event["type"] == "executing":
                    # SPARK_TIMING=1: how long each node of the workflow ran
                    if running is not None:
                        stages.append((workflow[running]["class_type"], time.time() - since))
                    running, since = (str(data["node"]) if data.get("node") is not None else None), time.time()
                    if running is None:
                        break
                elif event["type"] in ("execution_error", "execution_interrupted", "execution_success"):
                    break
    if os.environ.get("SPARK_TIMING") == "1":
        print("[tensorfold] stages: " + ", ".join(f"{name} {took:.1f} s" for name, took in stages if took >= 0.05), flush=True)
    return prompt_id


def render(workflow: dict, sampler: str) -> tuple[dict, float]:
    """Run a workflow; returns (the saved file's record, seconds)."""

    started = time.time()
    try:
        prompt_id = asyncio.run(run(workflow, sampler))
    except KeyboardInterrupt:
        call("/interrupt", {})
        raise
    while True:
        history = call(f"/history/{prompt_id}")
        if prompt_id in history and history[prompt_id].get("status", {}).get("completed") is not None:
            break
        time.sleep(1)
    record = history[prompt_id]
    if record["status"].get("status_str") != "success":
        errors = [m[1] for m in record["status"].get("messages", []) if m[0] == "execution_error"]
        detail = errors[0].get("exception_message", "").strip() if errors else json.dumps(record["status"])[:600]
        sys.exit(f"ComfyUI failed: {detail} (log: {PREFIX}/comfy.log)")
    # the files the save nodes wrote: a loader of a reference video or sound lists its input file among the outputs too
    found = [item for output in record["outputs"].values() for items in output.values()
             for item in (items if isinstance(items, list) else []) if isinstance(item, dict) and item.get("filename")
             and item.get("type", "output") == "output"]
    # the clip or the picture first, then the lossless sound track (save_sound) when the workflow has one
    found.sort(key=lambda item: item["filename"].endswith(".flac"))
    if not found or found[0]["filename"].endswith(".flac"):
        sys.exit("ComfyUI finished without writing a file")
    return Saved(found[0], next((item for item in found if item["filename"].endswith(".flac")), None)), time.time() - started


class Saved(dict):
    """The record of the clip or picture a workflow saved; ``sound`` is the record of its lossless sound track."""

    def __init__(self, item: dict, sound: dict | None = None):
        super().__init__(item)
        self.sound = sound



def fetch(item: dict, target: Path) -> None:
    query = urllib.parse.urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output")})
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL + "/view?" + query, timeout=600) as reply:
        target.write_bytes(reply.read())


def installed(kind: str, name: str) -> bool:
    """Whether ComfyUI lists a model file (it reads its folders again on every call)."""

    loader, field = {"unet": ("UNETLoader", "unet_name"), "lora": ("LoraLoaderModelOnly", "lora_name")}[kind]
    return name in call(f"/object_info/{loader}")[loader]["input"]["required"][field][0]


def forget_uploads() -> None:
    """Take this run's reference videos and sounds out of ComfyUI's input folder again (pictures stay)."""

    for name in UPLOADED:
        if name.endswith((".mp4", ".wav")):
            (PREFIX / "ComfyUI" / "input" / name).unlink(missing_ok=True)


def stopped() -> str:
    """Why the ComfyUI server ended, when its log says the memory guard (memory_guard.py watch) stopped it."""

    try:
        tail = (PREFIX / "comfy.log").read_text(errors="replace")[-4000:].splitlines()
    except OSError:
        return ""
    for line in reversed(tail):
        if "stopping ComfyUI" in line and ("memory floor:" in line or "free memory is" in line):
            return ("the render was stopped to protect the machine: " + line.split("] ", 1)[-1].replace("memory floor: ", "").strip()
                    + ". Use a shorter clip, a shorter or fewer reference videos, or a smaller size; a LoRA clip needs more memory than one without.")
    return ""
