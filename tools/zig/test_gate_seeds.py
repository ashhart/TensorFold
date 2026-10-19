"""Release phases retain exact comparisons with explicit keys, including old stored resume bodies."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


class Client:
    model = "fixture"

    def __init__(self):
        self.bodies = []

    def send(self, phase, name, body, **fields):
        self.bodies.append(body)
        return {"error": None, "cached_tokens": 0, "content": "fixed", "reasoning": "",
                "token_sha": "fixed", "stream": fields.get("stream", 0), "aggregate_tps": 1,
                "reply_model": self.model}

    @property
    def rows(self):
        return [{"reply_model": self.model}]

    def send_many(self, phase, name, bodies, **fields):
        return [self.send(phase, name, body, stream=i) for i, body in enumerate(bodies)]


class GateSeeds(unittest.TestCase):
    def test_all_phases_pin_omitted_keys_and_keep_named_keys(self):
        work = types.ModuleType("qual_work")
        body = {"temperature": 0.7, "max_tokens": 64}
        work.CODE, work.SEEDS = ["prompt"], [7]
        for name in ("warmup", "serial9", "resume_prefixes", "ladder"):
            setattr(work, name, lambda *a: [("fixture", body)])
        work.cells = lambda *a, **kw: [("sampled", "p0-s7", {**body, "seed": 7})]
        work.sweep_body = lambda *a: {**body, "seed": None}
        work.follow_up = lambda *a: body
        client_module = types.ModuleType("qual_client")

        def pair(client, phase, name, request, checks):
            one = client.send(phase, name, request)
            two = client.send(phase, name, {**request, "draft": False})
            self.assertEqual(one["token_sha"], two["token_sha"])
            return one, two

        client_module.pair = pair
        client_module.same = lambda one, two: one["token_sha"] == two["token_sha"]
        modules = {"qual_work": work, "qual_client": client_module, "qual_run": types.ModuleType("qual_run")}
        with patch.dict(sys.modules, modules), patch.object(sys, "path", [str(HERE), *sys.path]):
            spec = importlib.util.spec_from_file_location("seed_gate_visit", HERE / "gate_visit.py")
            gate = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gate)
        plan = {"phases": ["serial9", "cells", "sweep", "resume", "ladder"], "cell_tokens": [64],
                "sweep": [2], "sweep_reps": 1, "ladder": [1], "ladder_tokens": 64, "max_tokens": 16}
        client = Client()
        with tempfile.TemporaryDirectory() as folder:
            canonical = Path(folder) / "resume.json"
            gate.phases(client, plan, 1, canonical, [])
            # A preexisting canonical file may have omitted its key too.
            stored = json.loads(canonical.read_text())
            for entry in stored:
                entry["body"].pop("seed", None)
            canonical.write_text(json.dumps(stored))
            gate.phases(client, plan, 2, canonical, [])
        self.assertTrue(client.bodies)
        self.assertTrue(all(b.get("seed") in (7, 1234) for b in client.bodies))
        self.assertTrue(any(b["seed"] == 7 for b in client.bodies))
        self.assertNotIn("seed", body)


if __name__ == "__main__":
    unittest.main()
