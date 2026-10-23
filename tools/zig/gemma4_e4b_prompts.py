"""The 15 chats of the Gemma 4 E4B references (short, code, French, long documents, robot tool loops), all synthetic."""
# The long documents are slices of the model card; the fictional robot and its depth grids are generated.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TOOLS = [
    {"type": "function", "function": {"name": "robot_move", "description": "Walk: velocity twist held until replaced. vx forward m/s (>=0.3 to walk), vy left m/s, vyaw turn rad/s (positive = left).", "parameters": {"type": "object", "properties": {"vx": {"type": "number"}, "vy": {"type": "number"}, "vyaw": {"type": "number"}, "seconds": {"type": "number", "description": "how long to hold"}}, "required": ["vx", "vy", "vyaw"]}}},
    {"type": "function", "function": {"name": "robot_do", "description": "Run a one-shot skill.", "parameters": {"type": "object", "properties": {"skill": {"type": "string", "enum": ["sit_toggle", "roulade"]}}, "required": ["skill"]}}},
    {"type": "function", "function": {"name": "robot_look", "description": "Look at a point in the trunk frame, metres (x forward, y left, z up).", "parameters": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}, "required": ["x", "y", "z"]}}},
    {"type": "function", "function": {"name": "robot_sound", "description": "Play a voice tag.", "parameters": {"type": "object", "properties": {"tag": {"type": "string", "enum": ["alarm", "greet", "inquire", "peck", "chirp", "coo", "wheee"]}}, "required": ["tag"]}}},
    {"type": "function", "function": {"name": "robot_stop", "description": "Stop moving now.", "parameters": {"type": "object", "properties": {}}}},
]
SYSTEM = "You are Microduck, a small duck robot. Every message gives you the 8x8 depth grid (decimetres, row 0 = top). Decide with tools; keep replies under 15 words."


def frame(seed: int, wall: bool) -> str:
    rng = np.random.default_rng(seed)
    base = rng.integers(8, 30, size=(8, 8))
    if wall:
        base[2:7, 2:6] = rng.integers(1, 3, size=(5, 4))
    return "DEPTH " + " ".join(str(int(v)) for v in base.reshape(-1))


def call(name: str, args: dict, i: int) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": args}}]}


def robot(name: str, user: list[dict]) -> dict:
    return {"name": name, "messages": [{"role": "system", "content": SYSTEM}] + user, "tools": TOOLS}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path, help="the checkpoint directory (its README.md supplies the long documents)")
    p.add_argument("output", type=Path)
    args = p.parse_args()
    card = (args.model / "README.md").read_text()
    prompts = [
        {"name": "chat_short", "messages": [{"role": "user", "content": "In two sentences, why is the sky blue?"}]},
        {"name": "code", "messages": [{"role": "user", "content": "Write a Python function that merges two sorted lists into one sorted list, with a short docstring."}]},
        {"name": "french", "messages": [{"role": "system", "content": "Tu réponds toujours en français, de façon concise."}, {"role": "user", "content": "Explique la différence entre un processus et un thread, avec un exemple."}]},
        {"name": "doc1500", "messages": [{"role": "user", "content": card[3000:9500] + "\n\nQuestion: according to the text above, what is this model post-trained for? Answer in one sentence."}]},
        {"name": "doc4096", "messages": [{"role": "user", "content": card[3000:21000] + "\n\nQuestion: list three facts stated in the text above."}]},
        {"name": "multiturn", "messages": [{"role": "user", "content": "Give me a name for a pet duck."}, {"role": "assistant", "content": "How about Biscuit?"}, {"role": "user", "content": "Something more serious, and explain the choice."}]},
        robot("duck_wall", [{"role": "user", "content": frame(1, True)}]),
        robot("duck_free", [{"role": "user", "content": frame(2, False)}]),
        robot("duck_greet", [{"role": "user", "content": "A person waves at you. " + frame(3, False)}]),
        robot("duck_sit", [{"role": "user", "content": "Please sit down. " + frame(4, False)}]),
        robot("duck_after_move", [{"role": "user", "content": frame(5, False)}, call("robot_move", {"vx": 0.35, "vy": 0.0, "vyaw": 0.0, "seconds": 1.0}, 1), {"role": "tool", "tool_call_id": "c1", "content": "ok"}, {"role": "user", "content": frame(6, True)}]),
        robot("duck_look", [{"role": "user", "content": "Look at the red ball on your left, about one metre away. " + frame(7, False)}]),
        robot("duck_loop", [{"role": "user", "content": frame(8, False)}, call("robot_move", {"vx": 0.3, "vy": 0.0, "vyaw": 0.0, "seconds": 1.0}, 1), {"role": "tool", "tool_call_id": "c1", "content": "ok"}, {"role": "user", "content": frame(9, False)}, call("robot_move", {"vx": 0.3, "vy": 0.0, "vyaw": 0.0, "seconds": 1.0}, 2), {"role": "tool", "tool_call_id": "c2", "content": "ok"}, {"role": "user", "content": frame(10, True)}]),
        {"name": "tool_chat", "messages": [{"role": "user", "content": "What time is it in Tokyo?"}], "tools": [{"type": "function", "function": {"name": "get_time", "description": "Current time in a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]},
        {"name": "json_out", "messages": [{"role": "user", "content": "Return a JSON object with keys name, legs and sound for a duck. Only JSON."}]},
        {"name": "think", "enable_thinking": True, "messages": [{"role": "user", "content": "A farm has ducks and cows: 20 heads, 56 legs. How many of each?"}]},
    ]
    args.output.write_text(json.dumps(prompts, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(prompts)} prompts -> {args.output}")


if __name__ == "__main__":
    main()
