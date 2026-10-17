"""ComfyUI plumbing of the Library actions (spark_workflows.py): the server, LoRAs, running a workflow, shared nodes."""
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import spark_generate as sg

SAM = "sam3.1_multiplex_fp16.safetensors"
NODES = {"turnaround": ("H3ContactSheet", "H3ContactSheetDecode"), "upscale": ("H3VideoUpsampler",),
         "mask": ("SAM3_Detect", "SAM3_VideoTrack", "SAM3_TrackToMask", "H3RelayGreenMask"),
         "remove": ("SAM3_Detect", "SAM3_VideoTrack", "SAM3_TrackToMask", "H3RelayGreenMask", "H3RelayPersonRemover")}


def say(text: str) -> None:
    print("[tensorfold] " + text, flush=True)


def listed(loader: str, field: str) -> list:
    return sg.call(f"/object_info/{loader}")[loader]["input"]["required"][field][0]


# the upsampler's and relay's packs each patch H3's layout and clash in one server: PATCHED = pack and its pid
PATCHED = sg.PREFIX / "comfy.patched"
PACK = {"upscale": "upsampler", "remove": "relay"}


def comfy_pid() -> str:
    port = sg.URL.rsplit(":", 1)[-1]
    return subprocess.run(["pgrep", "-f", f"[m]ain.py --listen 127.0.0.1 --port {port}"], capture_output=True, text=True).stdout.split("\n")[0].strip()


def server(action: str) -> None:
    """ComfyUI up with nothing of another kind loaded, and the action's nodes and files present."""

    last = sg.STATE.read_text().strip() if sg.STATE.exists() else ""
    other = PATCHED.read_text().split() if PATCHED.exists() else []
    clash = action in PACK and len(other) == 2 and other[0] != PACK[action] and other[1] == comfy_pid()
    if sg.up() and (last.startswith("video") or clash):
        # as for a reference clip: TensorFold's engine keeps its 8-bit weights until the server is gone
        say("restarting ComfyUI: this starts from a server with nothing loaded" if not clash else
            "restarting ComfyUI: the upscaler's and the remover's node packs do not share one server")
        subprocess.run(["bash", str(sg.HERE / "comfy.sh"), "stop"], check=False)
        for _ in range(60):
            if not sg.up():
                break
            time.sleep(1)
        time.sleep(5)
    sg.ensure_server("workflow-" + action)
    have = sg.call("/object_info")
    missing = [n for n in NODES[action] if n not in have]
    if missing:
        sys.exit(f"ComfyUI has no node {', '.join(missing)}: run `bash spark/setup-workflows.sh`, then `bash spark/comfy.sh stop`")
    if sg.REF2VA["bf16"] not in listed("UNETLoader", "unet_name"):
        sys.exit("MiniMax H3 Ref2VA is not installed: bash oneshot-setup-spark.sh --ref2va")
    if action in PACK:
        PATCHED.write_text(f"{PACK[action]} {comfy_pid()}")


def lora_name(spec: str) -> tuple[str, float]:
    """A LoRA file as ComfyUI's loader names it (linked into its loras folder), and its strength."""

    path, strength = spec, 1.0
    head, _, tail = spec.rpartition(":")
    try:
        if head:
            path, strength = head, float(tail)
    except ValueError:
        pass
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        sys.exit(f"no LoRA at {path}")
    link = sg.MODELS / "loras" / f"hub_{source.name}"
    if not link.exists() or link.resolve() != source:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.unlink(missing_ok=True)
        link.symlink_to(source)
    if link.name not in listed("LoraLoaderModelOnly", "lora_name"):
        sys.exit(f"ComfyUI does not list {link.name}")
    return link.name, strength


async def follow(workflow: dict, sampler: str, parts: int) -> str:
    """Queue a workflow, print `sampler`'s steps until it ends; `parts` inner samplers (`sampler.x.y`) count as one."""

    import aiohttp

    client = uuid.uuid4().hex
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(sg.URL.replace("http", "ws", 1) + f"/ws?clientId={client}", max_msg_size=0) as ws:
            prompt_id = sg.call("/prompt", {"prompt": workflow, "client_id": client})["prompt_id"]
            inner, last = [], None
            async for message in ws:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                event = json.loads(message.data)
                data = event.get("data", {})
                if data.get("prompt_id") not in (None, prompt_id):
                    continue
                node = str(data.get("node"))
                if event["type"] == "progress" and (node == sampler or node.startswith(sampler + ".")):
                    at, of = int(data["value"]), int(data["max"])
                    if parts > 1 and node != sampler:
                        if of < 2:
                            continue                      # a decode or an encode inside the window, not its passes
                        if node not in inner:
                            inner.append(node)
                        at, of = (inner.index(node)) * of + at, max(parts, len(inner)) * of
                    if (at, of) != last:
                        last = (at, of)
                        say(f"step {at}/{of}")
                elif event["type"] == "executing" and data.get("node") is None:
                    break
                elif event["type"] in ("execution_error", "execution_interrupted", "execution_success"):
                    break
    return prompt_id


def execute(workflow: dict, sampler: str, parts: int = 1) -> tuple[dict, float]:
    """Run a workflow; returns ({node id: [saved file records]}, seconds)."""

    started = time.time()
    try:
        prompt_id = asyncio.run(follow(workflow, sampler, parts))
    except KeyboardInterrupt:
        sg.call("/interrupt", {})
        raise
    while True:
        history = sg.call(f"/history/{prompt_id}")
        if prompt_id in history and history[prompt_id].get("status", {}).get("completed") is not None:
            break
        time.sleep(1)
    record = history[prompt_id]
    if record["status"].get("status_str") != "success":
        errors = [m[1] for m in record["status"].get("messages", []) if m[0] == "execution_error"]
        detail = errors[0].get("exception_message", "").strip() if errors else json.dumps(record["status"])[:600]
        sys.exit(f"ComfyUI failed: {detail} (log: {sg.PREFIX}/comfy.log)")
    saved = {}
    for node, output in record["outputs"].items():
        saved[node] = [item for items in output.values() for item in (items if isinstance(items, list) else [])
                       if isinstance(item, dict) and item.get("filename") and item.get("type", "output") == "output"]
    return saved, time.time() - started


def take(item: dict, target: Path) -> None:
    """Fetch a saved file and remove ComfyUI's copy."""

    sg.fetch(item, target)
    (sg.PREFIX / "ComfyUI" / "output" / item.get("subfolder", "") / item["filename"]).unlink(missing_ok=True)


def loaders(w: dict, vae: str | None = None) -> None:
    w["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": sg.REF2VA["bf16"], "weight_dtype": "default"}}
    w["3"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": sg.H3_TEXT, "type": "minimax", "device": "default"}}
    w["4"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae or sg.H3_VAE}}
    w["5"] = {"class_type": "VAELoader", "inputs": {"vae_name": sg.H3_AUDIO_VAE}}


def still_vae() -> str:
    """The fp16 video decoder for still pictures when it is installed (the 8-bit one is the clips' default)."""

    wanted = os.environ.get("SPARK_VAE") or "minimax_h3_video_vae_fp16.safetensors"
    return wanted if wanted in listed("VAELoader", "vae_name") else sg.H3_VAE


def mask_nodes(w: dict, source: str, person: str, expand: int) -> None:
    """SAM 3.1 finds `person` in the first frame and follows it; the relay's node paints it green (nodes 20 to 27)."""

    w.update({
        "20": {"class_type": "LoadVideo", "inputs": {"file": source}},
        "21": {"class_type": "GetVideoComponents", "inputs": {"video": ["20", 0]}},
        "22": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM}},
        "23": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["22", 1], "text": person}},
        "24": {"class_type": "ImageFromBatch", "inputs": {"image": ["21", 0], "batch_index": 0, "length": 1}},
        # the author's settings: detection threshold 0.4, tracking as in their workflow
        "25": {"class_type": "SAM3_Detect", "inputs": {"model": ["22", 0], "image": ["24", 0], "conditioning": ["23", 0]}},
        "26": {"class_type": "SAM3_VideoTrack", "inputs": {"images": ["21", 0], "model": ["22", 0], "initial_mask": ["25", 0],
                                                         "conditioning": ["23", 0]}},
        "27": {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["26", 0]}},
        "28": {"class_type": "H3RelayGreenMask", "inputs": {"images": ["21", 0], "mask": ["27", 0], "expand_pixels": expand}},
    })
    # widget values in the order the author's workflow saved them, by each node's own input names
    info = sg.call("/object_info")
    for node, values in (("25", [0.4, 2, False]), ("26", [0.5, 1, 12]), ("27", [""])):
        spec = info[w[node]["class_type"]]["input"]
        names = [n for group in ("required", "optional") for n, d in spec.get(group, {}).items()
                 if n not in w[node]["inputs"] and not (isinstance(d[0], str) and d[0].isupper() and d[0] not in ("INT", "FLOAT", "STRING", "BOOLEAN"))]
        for name, value in zip(names, values):
            w[node]["inputs"][name] = value


def remover_nodes(w: dict, lora: tuple[str, float], anchor: str, frames_from: list, args, window: int, seed: int) -> None:
    """The author's graph from the green clip on: nodes 2 and 30 to 33. `frames_from` is the green-masked picture."""

    loaders(w, still_vae())
    w.update({
        "2": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0]}},
        "29": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": lora[0], "strength_model": lora[1]}},
        "30": {"class_type": "LoadImage", "inputs": {"image": anchor}},
        "31": {"class_type": "H3RelayPersonRemover", "inputs": {
            "model": ["29", 0], "clip": ["3", 0], "video_vae": ["4", 0], "audio_vae": ["5", 0], "masked_source": frames_from,
            "clean_first_frame": ["30", 0], "fps": 24.0, "prompt": args.prompt, "seed": seed, "steps": args.steps,
            "window_frames": str(window), "window_controls": "{}"}},
    })
    spec = sg.call("/object_info/MiniMaxH3SigmaShift")["MiniMaxH3SigmaShift"]["input"]["required"]
    for name, value in zip([n for n in spec if n != "model"], (12, 3)):   # video and audio shift, as the author set them
        w["2"]["inputs"][name] = value
