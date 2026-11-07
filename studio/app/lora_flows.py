"""What a hub LoRA flow (character-swap, lineart, keyframe, slider) needs of a clip, and its trained prompt style."""

from __future__ import annotations

import json
from pathlib import Path

PRESETS = Path(__file__).with_name("lora-presets.json")

FLOWS = {
    "character-swap": {
        "videos": (1, 1), "pictures": (1, 9), "plain": False, "turbo": True,
        "asks": ["The source clip as Video 1 (4 to 5 seconds works best)", "A picture of the new character as Picture 1"],
        "hint": "Under “Who or what it shows” of Video 1, say who is replaced: the man in the purple shirt.",
    },
    "lineart": {
        "videos": (1, 1), "pictures": (0, 0), "plain": True, "turbo": True, "lineart": True,
        "asks": ["A line-art video as Video 1: “Make line art of Video 1” draws one from any clip"],
        "hint": "Describe the finished picture in Scene and Action: who, hair, clothes, place, light.",
    },
    "keyframe": {
        "videos": (0, 0), "pictures": (1, 2), "plain": True, "turbo": False,
        "asks": ["One drawing (the current key) or two (the first drawing and the end extreme) as pictures"],
        "hint": "The adapters hold a hand-drawn line on the full model only: 20 passes, not Turbo.",
    },
    "slider": {"videos": (0, 3), "pictures": (0, 9), "plain": False, "turbo": True, "asks": [], "hint": ""},
}


def flow_of(lora_id: str) -> str:
    """The flow a built-in preset names, or "" for a LoRA with none (it rides on any clip)."""

    try:
        found = next((r for r in json.loads(PRESETS.read_text()) if r.get("id") == lora_id), None)
    except (OSError, ValueError):
        found = None
    flow = (found or {}).get("flow") or ""
    return flow if flow in FLOWS else ""


def picked(loras) -> list[tuple[str, str]]:
    """[(LoRA id, flow)] of a job's or a form's `loras` that name a flow."""

    out = []
    for item in loras or []:
        lora_id = str((item or {}).get("id") or "")
        flow = flow_of(lora_id)
        if flow:
            out.append((lora_id, flow))
    return out


def fits(loras, pictures: int, videos: int, turbo: bool) -> str | None:
    """None when the clip holds what every picked LoRA needs, else one sentence saying what is missing."""

    for lora_id, flow in picked(loras):
        rule, name = FLOWS[flow], lora_id.replace("-", " ").title()
        for kind, count in (("videos", videos), ("pictures", pictures)):
            low, high = rule[kind]
            word = "reference video" if kind == "videos" else "reference picture"
            if count < low:
                return f"{name} needs: " + "; ".join(rule["asks"]) + " (References step)"
            if count > high:
                return f"{name} takes {'no' if high == 0 else 'at most ' + str(high)} {word}{'' if high == 1 else 's'}"
        if turbo and not rule["turbo"]:
            return f"{name} runs on the full model: choose 20 passes, not the Turbo adapter"
    return None


def plain(loras) -> bool:
    """Whether the prompt goes to the model as written (a LoRA trained on plain captions)."""

    return any(FLOWS[flow]["plain"] for _, flow in picked(loras))


def sentence(text: str) -> str:
    text = " ".join(str(text or "").split()).rstrip(".")
    return text


def compose(loras, fields: dict) -> str | None:
    """The prompt in the caption style of the first picked LoRA that has one; None when none has (reference prompt)."""

    flows = [flow for _, flow in picked(loras)]
    scene, action, camera = sentence(fields.get("scene")), sentence(fields.get("action")), sentence(fields.get("camera"))
    sound = sentence(fields.get("soundscape"))
    if "character-swap" in flows:
        who = sentence((fields.get("video_subjects") or [""])[0]) or "the person"
        new = sentence((fields.get("subjects") or [""])[0])
        text = (f"Replace only {who} in <Video 1> with the character in <Picture 1>"
                + (f", {new}" if new else "") + ". Keep the replacement character's identity, outfit, and art style from "
                "<Picture 1>. Preserve the source video's camera, background, lighting, objects, and all other people. "
                "Match the target person's position, scale, pose, and movement. Do not show the reference sheet or its background.")
        return text + (f" {sound}." if sound else "")
    if "lineart" in flows:
        parts = [part for part in (scene, action) if part]
        return (". ".join(parts) + ".") if parts else "Generate anime videos from line art outlines."
    if "keyframe" in flows:
        static = sentence(fields.get("static") or camera) or "the framing and the ground line hold"
        return (f"Create the next keyframe in the sequence. SCENE: {scene or 'a single figure, clean line on white'}. "
                f"MOVEMENT: {action or 'the figure carries the action through to its next extreme'}. STATIC: {static}.")
    return None


def public(lora_id: str) -> dict | None:
    """What the hub's card and the wizard show for a preset's flow."""

    flow = flow_of(lora_id)
    if not flow:
        return None
    rule = FLOWS[flow]
    return {"flow": flow, "asks": rule["asks"], "hint": rule["hint"], "turbo": rule["turbo"], "lineart": bool(rule.get("lineart")),
            "videos": list(rule["videos"]), "pictures": list(rule["pictures"])}
