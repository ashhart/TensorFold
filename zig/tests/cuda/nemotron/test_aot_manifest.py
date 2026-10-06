#!/usr/bin/env python3
"""CPU test for the capture's manifest step, no GPU and no model: a fake Triton cache and a launch dump
go through tools/zig/triton_aot_manifest.py and aot_pack.py, and the kernel set's entry is checked.
Usage: python3 -B test_aot_manifest.py"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
MANIFEST = REPO / "tools" / "zig" / "triton_aot_manifest.py"
PACK = Path(__file__).resolve().parent / "aot_pack.py"
HASH = "b" * 64
CUBIN = b"FAKE-CUBIN-BYTES"
MOUNT = "/aot/triton"


def fake_cache(cache: Path) -> dict:
    """One kernel's Triton cache (metadata JSON, PTX, cubin) and the Recorder's launch dump for it."""

    kdir = cache / HASH
    kdir.mkdir(parents=True)
    (kdir / "fake_kernel.cubin").write_bytes(CUBIN)
    (kdir / "fake_kernel.ptx").write_text(
        ".entry fake_kernel(\n .param .u64 p0,\n .param .u32 n,\n .param .u64 global_scratch,\n"
        " .param .u64 profile_scratch\n) .reqntid 128, 1, 1\n{\n ret;\n}\n")
    meta = {"name": "fake_kernel", "num_warps": 4, "num_ctas": 1, "shared": 0, "global_scratch_size": 0,
            "global_scratch_align": 1, "profile_scratch_size": 0, "profile_scratch_align": 1,
            "launch_pdl": False, "hash": HASH, "triton_version": "test", "target": {"arch": 121}}
    (kdir / "__grp__fake_kernel.json").write_text(json.dumps(meta) + "\n")
    files = {leaf: f"{MOUNT}/{HASH}/{leaf}"
             for leaf in ("__grp__fake_kernel.json", "fake_kernel.cubin", "fake_kernel.ptx")}
    info = {"name": "fake_kernel", "function": "fakemod.FakeKernel",
            "source": {"file": "tensorfold/families/nemotron_h/cuda/kernels.py", "line": 10},
            "params": ["p0", "n", "BLOCK"], "signature": {"p0": "*i8", "n": "i32", "BLOCK": "constexpr"},
            "constexprs": {"BLOCK": 64}, "attrs": {"p0": [["tt.divisibility", 16]]},
            "metadata": {"num_warps": 4}, "files": files, "n_regs": 24, "n_spills": 0}
    return {"generator": "test", "written_unix": 0, "kernels": {HASH: info},
            "phases": {"startup": {HASH: 3}}, "grids": {HASH: [[128, 1, 1]]},
            "sites": {HASH: ["kernels.py:10 fake_launch"]}, "log": []}


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        cache, out = work / "cache", work / "out"
        cache.mkdir()
        out.mkdir()
        launches = out / "launches.json"
        launches.write_text(json.dumps(fake_cache(cache)) + "\n")

        manifest = out / "manifest.json"
        r = subprocess.run([sys.executable, "-B", str(MANIFEST), "--cache", str(cache),
                            "--launches", str(launches), "--mount", MOUNT, "--out", str(manifest)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout + r.stderr)
            return 1
        m = json.loads(manifest.read_text())
        assert m["generator"] == "tools/zig/triton_aot_manifest.py", m["generator"]
        k = m["kernels"][0]
        assert k["hash"] == HASH and k["cubin"] == f"{HASH}/fake_kernel.cubin", k
        assert k["cubin_sha256"] == hashlib.sha256(CUBIN).hexdigest()
        assert [p["name"] for p in k["abi"]] == ["p0", "n", "global_scratch", "profile_scratch"], k["abi"]
        assert k["launches"] == {"startup": 3} and k["grids"] == [[128, 1, 1]]

        jit = out / "jit.json"
        jit.write_text(json.dumps({"fakemod.FakeKernel": {"params": ["p0", "n", "BLOCK"],
                                                          "do_not_specialize": [], "no_align": []}}) + "\n")
        packed = work / "packed"
        r = subprocess.run([sys.executable, "-B", str(PACK), "--manifest", str(manifest),
                            "--cache", str(cache), "--jit", str(jit), "--out", str(packed)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout + r.stderr)
            return 1
        aot = json.loads((packed / "aot.json").read_text())
        e = aot["kernels"][0]
        assert e["fn"] == "fake_kernel" and e["hash"] == HASH and e["num_warps"] == 4, e
        assert e["params"] == [{"name": "p0", "type": "*i8", "div16": True, "nospec": False},
                               {"name": "n", "type": "i32", "div16": False, "nospec": False}], e["params"]
        assert e["consts"] == {"BLOCK": {"int": 64}}, e["consts"]
        assert (packed / "cubins" / f"{HASH}.cubin").read_bytes() == CUBIN
        print("test_aot_manifest: 1 kernel through the manifest and the pack, entry checked")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
