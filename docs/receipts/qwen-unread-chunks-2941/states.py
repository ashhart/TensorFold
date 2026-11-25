"""Kept-state and K/V digests at a prompt-cache mark, from the disk tier's own files."""
import hashlib
import json
import struct
import sys
from pathlib import Path

MAGIC = 0x42524654
BLOCK_TOKENS = 256
WORDS = 16
TRAILER = struct.Struct("<IIIIQQQQ32sIIQ")


def words(*values):
    return b"".join(struct.pack("<I", value) for value in values)


def edges(digest, spans, starts, lo, hi):
    inside = [span for span in spans if span[0] < hi and span[1] > lo]
    digest.update(words(len(inside)))
    for span in inside:
        digest.update(words(max(span[0], lo), min(span[1], hi)))
    inside = [start for start in starts if lo <= start < hi]
    digest.update(words(len(inside)))
    for start in inside:
        digest.update(words(start))


def next_start(starts, at, end):
    return min([start for start in starts if at <= start < end] or [end])


def block_names(identity, tokens, spans, starts, at):
    seeded = hashlib.sha256()
    seeded.update(b"tensorfold rows 1")
    seeded.update(words(identity & 0xFFFFFFFF, identity >> 32, BLOCK_TOKENS, 0, 1))
    chain = seeded.digest()
    names = []
    for lo in range(0, at, BLOCK_TOKENS):
        hi = lo + BLOCK_TOKENS
        digest = hashlib.sha256()
        digest.update(chain)
        digest.update(struct.pack(f"<{BLOCK_TOKENS}I", *tokens[lo:hi]))
        edges(digest, spans, starts, lo, hi)
        chain = digest.digest()
        named = hashlib.sha256()
        named.update(chain)
        named.update(words(0, 0))  # window, lookahead
        named.update(words(next_start(starts, hi, at)))
        names.append(named.digest())
    return names


def parse(path):
    data = path.read_bytes()
    magic, version, kind, flags, payload, payload_hash, meta_bytes, meta_hash, name, at, block, reserved = TRAILER.unpack(data[-TRAILER.size:])
    if magic != MAGIC or version != 1 or kind != 2 or block != BLOCK_TOKENS or payload + meta_bytes + TRAILER.size != len(data):
        return None
    counts = struct.unpack_from("<III", data, payload)
    offset = payload + 12
    tokens = struct.unpack_from(f"<{counts[0]}I", data, offset)
    offset += 4 * counts[0]
    spans = [struct.unpack_from("<II", data, offset + 8 * i) for i in range(counts[1])]
    offset += 8 * counts[1]
    starts = struct.unpack_from(f"<{counts[2]}I", data, offset)
    return {"name": name, "at": at, "payload": data[:payload], "tokens": tokens, "spans": spans, "starts": starts}


def digest_state(payload):
    return hashlib.sha256(payload[WORDS:]).hexdigest()


def digest_kv(root, identity, state):
    digest = hashlib.sha256()
    names = block_names(identity, state["tokens"], state["spans"], state["starts"], state["at"])
    for name in names:
        data = (root / (name[:16].hex() + ".blk")).read_bytes()
        payload = TRAILER.unpack(data[-TRAILER.size:])[4]
        digest.update(data[:payload])
    return digest.hexdigest(), len(names)


def main():
    root, marks = Path(sys.argv[1]), [int(mark) for mark in sys.argv[2:]]
    out = {str(mark): [] for mark in marks}
    for path in sorted(root.rglob("*.st")):
        state = parse(path)
        if state is None or state["at"] not in marks:
            continue
        identity = int(path.parent.name, 16)
        blocks, count = digest_kv(path.parent, identity, state)
        out[str(state["at"])].append({"state": path.stem, "blocks": count, "state_sha256": digest_state(state["payload"]), "kv_sha256": blocks})
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
