"""Record KV handoff test vectors from MCDMA's own Python producer (integrations/vllm/mcdma_kv), for the Zig fabric."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPLY = 4096


class Tensor:
    """A cache tensor of deterministic bytes; byte i is (i * 37 + seed) % 256."""

    def __init__(self, shape, itemsize, seed):
        self.shape, self.itemsize, self.seed = tuple(shape), itemsize, seed
        count = itemsize
        for size in shape:
            count *= size
        self.data = bytes((i * 37 + seed) % 256 for i in range(count))

    def element_size(self):
        return self.itemsize

    def rows(self, block_rows):
        """index_select(0, rows).contiguous() for a block-first tensor."""
        inner = len(self.data) // self.shape[0]
        return b"".join(self.data[r * inner:(r + 1) * inner] for r in block_rows)


class Mailbox:
    """The reply half as MCDMA's own tests stand it in: replies captured in order."""

    def __init__(self):
        self.max_reply = REPLY
        self.area = bytearray(REPLY)
        self.replies = []

    def reply(self, seq, parts):
        self.replies.append(b"".join(bytes(part) for part in parts))

    def reply_area(self):
        return memoryview(self.area)

    def publish(self, seq, length):
        self.replies.append(bytes(self.area[:length]))


def session(mcdma: Path) -> dict:
    """Drive MCDMA's Responder through every request kind and outcome; return requests, replies and inputs."""
    sys.path.insert(0, str(mcdma / "integrations" / "vllm"))
    from mcdma_kv import wire
    from mcdma_kv.export import Export, LayerPages
    from mcdma_kv.responder import ExportTable, Responder

    specs = [{"shape": [10, 2, 16, 16], "itemsize": 2, "seed": 11, "rows": [4, 5, 6, 1, 9], "index": 0,
              "dims": ["block", "head", "token", "kv_head_dim"], "dtype": "bfloat16"},
             {"shape": [6, 16, 72], "itemsize": 2, "seed": 5, "rows": [2, 0], "index": 7,
              "dims": ["block", "token", "latent"], "dtype": "float16"}]
    tensors = [Tensor(s["shape"], s["itemsize"], s["seed"]) for s in specs]

    def layers():
        return [LayerPages(index=0, tensor=tensors[0], block_dim=0, rows=specs[0]["rows"], dims=specs[0]["dims"],
                           dtype="bfloat16", heads=2, total_heads=4, head_size=8),
                LayerPages(index=7, tensor=tensors[1], block_dim=0, rows=specs[1]["rows"], dims=specs[1]["dims"],
                           dtype="float16", latent_size=64, rope_size=8)]

    def fill(export, frame, out):
        position, start, rows = export.frames[frame]
        layer = export.layers[position]
        data = layer.tensor.rows(layer.rows[start:start + rows])
        out[:len(data)] = data
        return len(data)

    first, second, failed = bytes(range(16)), bytes(range(100, 116)), bytes([7] * 16)
    tokens = [(i * 7919) % 160000 for i in range(40)]
    table = ExportTable(ttl_s=60, clock=lambda: 0.0)
    box = Mailbox()
    responder = Responder(box, table, fill, model="org/model-é", tp_rank=1, tp_size=2)
    events = []

    def ask(kind, handoff, frame=0, payload=b"", raw=None):
        message = raw if raw is not None else wire.pack(wire.Header(kind, handoff, frame=frame, nbytes=len(payload))) + payload
        responder.handle(len(events) + 1, memoryview(message))
        events.append({"request": message.hex(), "reply": box.replies[-1].hex()})

    def add(handoff, what):
        table.add(handoff, "req", what)
        events.append({"add": handoff.hex(), "failed": what if isinstance(what, str) else None})

    ask(wire.OPEN, first, payload=b'{"checksum": true}')
    add(first, Export("req-1", first, tokens, 0, 16, layers()))
    ask(wire.OPEN, first, payload=b'{"checksum": true}')
    frames = json.loads(bytes(wire.body(bytes.fromhex(events[-1]["reply"]), wire.unpack(bytes.fromhex(events[-1]["reply"])))))["frames"]
    for frame in range(frames + 1):
        ask(wire.PULL, first, frame=frame)
    ask(wire.MANIFEST, first)
    ask(wire.CLOSE, first)
    add(second, Export("req-2", second, tokens[:33], 16, 16, layers()))
    ask(wire.OPEN, second, payload=b'{"checksum": false}')
    ask(wire.PULL, second, frame=1)
    ask(wire.CLOSE, second)
    add(failed, "float8_e4m3fn KV caches cannot be exported")
    ask(wire.OPEN, failed, payload=b"{}")
    ask(wire.PULL, failed, frame=0)
    ask(0, b"", raw=b"\0" * 10)
    ask(0, b"", raw=b"XXXX" + b"\0" * 124)
    bad_kind = bytearray(wire.pack(wire.Header(wire.ACK, first)))
    bad_kind[6:8] = (42).to_bytes(2, "little")
    ask(0, b"", raw=bytes(bad_kind))
    ask(0, b"", raw=wire.pack(wire.Header(wire.OPEN, first, nbytes=8)))
    digests = [{"tokens": t, "sha256": wire.token_sha256(t)} for t in ([], [1, 70000], tokens)]
    header = wire.Header(wire.DATA, first, 3, 9, 2, wire.CHECKED, 16, 4, 8, 77)
    revision = subprocess.run(["git", "-C", str(mcdma), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"mcdma": revision, "reply_area": REPLY, "tensors": specs, "tokens": tokens, "events": events,
            "digests": digests, "header": wire.pack(header).hex()}


def main() -> int:
    """Write the vectors JSON from an MCDMA checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcdma", type=Path, default=ROOT / "build" / "mcdma")
    parser.add_argument("--out", type=Path, default=ROOT / "zig" / "tests" / "fabric_vectors.json")
    args = parser.parse_args()
    data = session(args.mcdma.resolve())
    args.out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"{len(data['events'])} events from MCDMA {data['mcdma'][:12]} -> {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
