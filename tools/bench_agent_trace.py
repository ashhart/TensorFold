"""Seeded agent traffic with measured assistant replies; tokenization sizes fixtures, never usage counters."""
import hashlib
import json
import random
import urllib.request

from bench_auth import authorize

SHAPES = ("agent_loops", "agent_cohorts")
PARAGRAPH = "alpha beta gamma delta epsilon. A public cache measurement uses fixed words and recorded steps.\n"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class Texts:
    """Build public text of a verified standalone token count using the selected server's tokenizer."""

    def __init__(self, base, model, timeout, api_key=None):
        self.base, self.model, self.timeout, self.key = base.rstrip("/"), model, timeout, api_key
        self.cache = {}

    def post(self, path, body):
        request = urllib.request.Request(self.base + path, json.dumps({"model": self.model, **body}).encode(),
                                         {"Content-Type": "application/json"})
        authorize(request, self.key)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read(16_777_217)
        if len(raw) > 16_777_216:
            raise ValueError("tokenizer_response_too_large")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("invalid_tokenizer_response")
        return value

    def tokenize(self, text):
        ids = self.post("/tokenize", {"prompt": text, "add_special_tokens": False}).get("tokens")
        if not isinstance(ids, list) or not ids or any(type(n) is not int or not 0 <= n <= 0xffffffff for n in ids):
            raise ValueError("invalid_tokenizer_ids")
        return ids

    def text(self, count, label="fixture"):
        key = count, label
        if key not in self.cache:
            repetitions = max(1, count // 8)
            for _ in range(16):
                source = label + "\n" + PARAGRAPH * repetitions
                ids = self.tokenize(source)
                if len(ids) >= count:
                    break
                repetitions *= 2
            else:
                raise ValueError("tokenizer_cannot_size_fixture")
            ids = ids[:count]
            text = self.post("/detokenize", {"tokens": ids}).get("prompt")
            if not isinstance(text, str) or self.tokenize(text) != ids:
                raise ValueError("fixture_token_roundtrip_failed")
            self.cache[key] = text, {"tokens": count, "text_sha256": digest(text), "ids_sha256": digest(ids)}
        return self.cache[key]


def prepare(texts, settings):
    prefix = settings["prefix_tokens"]
    phases = 2 if settings["trace"] == "agent_cohorts" else 1
    systems = [texts.text(prefix, f"{word} public {settings['trace']} seed {settings['seed']} cohort {i}")
               for i, word in enumerate(("amber", "zeta")[:phases])]
    if phases == 2 and systems[0][1]["ids_sha256"] == systems[1][1]["ids_sha256"]:
        raise ValueError("cohort_system_prefixes_not_distinct")
    initial = [texts.text(prefix + i * max(1, prefix // 8))[0] for i in range(3)]
    growth = texts.text(max(1, prefix // 16))[0]
    sizes = [max(1, prefix // 8), max(1, prefix // 2), prefix, max(1, prefix * 375 // 256)]
    one_offs = [texts.text(size)[0] for size in sizes]
    return systems, initial, growth, one_offs


def requests(settings, fixtures):
    """Yield one agent and two unique prompts; the caller appends each assistant reply to the yielded history."""
    systems, initial, growth, one_offs = fixtures
    conversations, rounds = settings["conversations"], settings["rounds"]
    rng = random.Random(settings["seed"])
    histories, previous_phase, serial = {}, None, 0
    for round_number in range(rounds):
        phase = int(settings["trace"] == "agent_cohorts" and round_number >= rounds // 2)
        if phase != previous_phase:
            histories = {}
            previous_phase = phase
        turn = round_number - (rounds // 2 if phase else 0)
        for slot in range(conversations):
            conversation = phase * conversations + slot
            if conversation not in histories:
                histories[conversation] = [{"role": "system", "content": systems[phase][0]},
                                           {"role": "user", "content": f"Conversation {conversation} begins.\n" + initial[slot % 3]}]
            else:
                histories[conversation].append({"role": "user", "content": f"Conversation {conversation}, step {turn}.\n" + growth})
            yield {"conversation": conversation, "round": round_number, "turn": turn, "kind": "agent"}, histories[conversation]
            for _ in range(2):
                name = f"oneoff-{serial:08d}"
                serial += 1
                yield {"conversation": name, "round": round_number, "turn": 0, "kind": "one_off"}, [
                    {"role": "user", "content": name + "\n" + rng.choice(one_offs)}]
