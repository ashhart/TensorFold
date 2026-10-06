"""Check native weight reads with synthetic files and bounded I/O faults."""
import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

SOURCE = Path(__file__).resolve().parents[2]
WORDS = 524288


def fixture(padding, family):
    values = array.array("I", range(1, WORDS + 1))
    if sys.byteorder != "little":
        values.byteswap()
    payload = values.tobytes()
    header = {}
    if padding:
        header["prefix"] = {"dtype": "U32", "shape": [padding // 4], "data_offsets": [0, padding]}
    if family == "flashnext-group":
        for shard in range(2):
            begin = padding + shard * len(payload) // 2
            name = f"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
            header[name] = {"dtype": "U32", "shape": [WORDS // 2], "data_offsets": [begin, begin + len(payload) // 2]}
    else:
        header["probe"] = {"dtype": "U32", "shape": [WORDS], "data_offsets": [padding, padding + len(payload)]}
    text = json.dumps(header).encode("utf-8")
    text += b" " * ((-len(text)) % 8)
    return struct.pack("<Q", len(text)) + text + bytes(padding) + payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=SOURCE / "zig-out/bin/tf-weight-read-check")
    parser.add_argument("--storage-directory", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--faults-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or not args.storage_directory.is_dir() or not args.binary.is_file():
        parser.error("Use existing paths and a positive repeat count.")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    with tempfile.TemporaryDirectory(prefix="tensorfold-loader-probe-", dir=args.storage_directory) as directory:
        for family in ["nemotron", "flashnext", "flashnext-group"]:
            for padding in [0, 64]:
                cases = [] if args.faults_only else [(mode, repeat) for repeat in range(1, args.repeats + 1) for mode in ["normal", "delayed"]]
                cases += [(mode, 1) for mode in ["eof", "eio", "eintr"]]
                for mode, repeat in cases:
                    name = f"{family}-{mode}-offset{padding}-{repeat}"
                    path = Path(directory) / ("tensorfold-loader-probe-" + name + ".safetensors")
                    with path.open("wb") as stream:
                        stream.write(fixture(padding, family))
                        stream.flush()
                        os.fsync(stream.fileno())
                    command = [str(args.binary.resolve()), family, str(path), mode]
                    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                    log = result.stdout + result.stderr
                    (args.output / (name + ".log")).write_text(log, encoding="utf-8")
                    lines = [line for line in log.splitlines() if line.startswith("{")]
                    record = json.loads(lines[-1]) if lines else {"output": log}
                    record.update(case=name, tensor_offset=padding, returncode=result.returncode)
                    records.append(record)
                    path.unlink()
                    print(json.dumps(record), flush=True)
    receipt = {"binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
               "loader_source_sha256": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in ["zig/src/core/checkpoint_metal.zig", "zig/src/families/flashnext/replay.zig"]},
               "fixture_payload_bytes": WORDS * 4, "delay_microseconds": 5000, "partial_read_limit": 32768,
               "page_cache": "uncontrolled", "temporary_files_removed": True, "cases": records,
               "passed": sum(x["returncode"] == 0 for x in records), "failed": sum(x["returncode"] != 0 for x in records)}
    (args.output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return int(receipt["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
