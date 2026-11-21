#!/usr/bin/env python3
"""Pipelined rounds (RFC #2005) receipt: token checks and tok/s cells, two boxes against the best single box."""
import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

PIPE_LINE = re.compile(r"^pipe (?:whole|rank 0): (\{.*\})\s*$", re.M)
DROP = "sync; echo 3 > /proc/sys/vm/drop_caches"


def sh(cmd, timeout=60):
    """A command's exit status and its output (stderr folded in); 127 when it does not exist."""
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def parse_ibv(text):
    """`ibv_devinfo -v` text: one entry a port, with its device, state, width, speed, MTU and link layer."""
    ports, dev, port = [], None, None
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"hca_id:\s*(\S+)", line)
        if m:
            dev = m.group(1)
            continue
        m = re.match(r"port:\s*(\d+)", line)
        if m:
            port = {"device": dev, "port": int(m.group(1))}
            ports.append(port)
            continue
        if port is None:
            continue
        for key in ("state", "active_width", "active_speed", "active_mtu", "link_layer", "phys_state"):
            m = re.match(rf"{key}:\s*(.+)$", line)
            if m:
                port[key] = m.group(1).strip()
    return ports


def parse_meminfo(text):
    """MemTotal and MemAvailable in GiB."""
    out = {}
    for key in ("MemTotal", "MemAvailable"):
        m = re.search(rf"^{key}:\s+(\d+) kB", text, re.M)
        if m:
            out[key.lower() + "_gib"] = round(int(m.group(1)) / 2**20, 2)
    return out


def parse_smi_cuda(text):
    """The driver's CUDA version from `nvidia-smi`'s banner."""
    m = re.search(r"CUDA (?:UMD )?Version:\s*([0-9.]+)", text)
    return m.group(1) if m else None


def iface_of(addr, ip_json):
    """The interface holding `addr`, from `ip -j addr`."""
    for link in ip_json:
        for a in link.get("addr_info", []):
            if a.get("local") == addr:
                return link.get("ifname")
    return None


def read_sys(path):
    """A sysfs value, or None (loopback and down links refuse `speed` with EINVAL)."""
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def facts(binary, repo, addr):
    """This host: memory, GPU, driver and CUDA, the link interface and its RDMA ports, the binary and its commit."""
    out = {"host": socket.gethostname(), "kernel": os.uname().release}
    out.update(parse_meminfo(Path("/proc/meminfo").read_text()))
    rc, smi = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    out["gpu"] = smi if rc == 0 else None
    rc, banner = sh(["nvidia-smi"])
    out["cuda_driver"] = parse_smi_cuda(banner) if rc == 0 else None
    rc, nvcc = sh(["nvcc", "--version"])
    if rc == 127:
        rc, nvcc = sh(["/usr/local/cuda/bin/nvcc", "--version"])
    out["nvcc"] = nvcc.splitlines()[-1] if rc == 0 and nvcc else None
    rc, ibv = sh(["ibv_devinfo", "-v"])
    out["ibv_ports"] = parse_ibv(ibv) if rc == 0 else []
    out["ibv_note"] = None if out["ibv_ports"] else (ibv or "ibv_devinfo not found")[:200]
    rc, ipj = sh(["ip", "-j", "addr"])
    ifname = iface_of(addr, json.loads(ipj)) if rc == 0 and addr else None
    out["link"] = {"addr": addr, "iface": ifname}
    if ifname:
        for key in ("speed", "mtu", "operstate"):
            out["link"][key] = read_sys(f"/sys/class/net/{ifname}/{key}")
        rc, eth = sh(["ethtool", ifname])
        m = re.search(r"Speed:\s*(\S+)", eth) if rc == 0 else None
        out["link"]["ethtool_speed"] = m.group(1) if m else None
    out["binary"] = {"path": str(binary), "md5": md5(binary) if Path(binary).exists() else None}
    rc, head = sh(["git", "-C", str(repo), "rev-parse", "HEAD"])
    out["commit"] = head if rc == 0 else None
    return out


def parse_pipe(text):
    """The tool's summary line: its JSON, or None when the run printed none."""
    m = PIPE_LINE.findall(text)
    return json.loads(m[-1]) if m else None


def compare(ref, got, streams, tokens):
    """Two token files: equal, or the first stream and position that differ."""
    if ref == got:
        return {"equal": True}
    n = 4 * tokens
    for s in range(streams):
        a, b = ref[s * n:(s + 1) * n], got[s * n:(s + 1) * n]
        if a != b:
            at = next((i for i in range(min(len(a), len(b)) // 4) if a[4 * i:4 * i + 4] != b[4 * i:4 * i + 4]), None)
            return {"equal": False, "stream": s, "token": at}
    return {"equal": False, "stream": None, "token": None, "lengths": [len(ref), len(got)]}


def prefix_equal(solo, many, tokens):
    """Stream 0 of a many-stream file against a one-stream run of the same prompt."""
    return solo[:4 * tokens] == many[:4 * tokens]


def verdicts(checks_):
    """Each token check's name and whether its tokens were equal."""
    return ", ".join(f"{k}={v.get('equal')}" for k, v in checks_.items() if isinstance(v, dict))


def table(rows, two="two boxes"):
    """The tok/s cells as markdown, two boxes (or two stages on loopback) against the best single box."""
    out = [f"| streams | box A alone | box B alone | {two}, k = 1 | {two}, k = 2 | k = 2 / best box |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        best = max([v for v in (r.get("whole_a"), r.get("whole_b")) if v] or [0])
        ratio = f"{r['ring_k2'] / best:.2f}x" if best and r.get("ring_k2") else "-"
        cells = [f"{r[k]:.1f}" if r.get(k) else "-" for k in ("whole_a", "whole_b", "ring_k1", "ring_k2")]
        out.append(f"| {r['streams']} | " + " | ".join(cells) + f" | {ratio} |")
    return "\n".join(out)


class Runner:
    """Starts the tool as one engine or as the ring's two ranks, each run its own processes, page cache dropped."""

    def __init__(self, a):
        self.a = a
        self.out = Path(a.out)
        self.secret = self.out / "secret"
        self.n = 0
        self.log = []

    def env(self, set_):
        e = {"TF_PIPE_STREAMS": set_["streams"], "TF_PIPE_K": set_["k"], "TF_PIPE_PROMPT": self.a.prompt,
             "TF_PIPE_TOKENS": set_["tokens"], "TF_PIPE_DRAFT": set_.get("draft", 0)}
        return {k: str(v) for k, v in e.items()}

    def ring_env(self, rank):
        peers = f"{self.a.addr}:{self.a.port},{self.a.peer_addr}:{self.a.port + 1}"
        return {"TF_STAGE_RANK": str(rank), "TF_STAGE_WORLD": "2", "TF_STAGE_PEERS": peers,
                "TF_STAGE_SECRET_FILE": str(self.secret if rank == 0 or self.a.loopback else self.a.peer_secret),
                "TF_STAGE_LAYOUT": self.a.layout, "TF_STAGE_TIMEOUT_S": "600"}

    def argv(self, binary, out):
        return [str(binary), self.a.model, "pipe", str(out)] + (["fp8"] if self.a.kv == "fp8" else [])

    def remote(self, env, argv):
        """The command line the peer runs: its own repo, binary and model paths."""
        sets = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        return f"cd {shlex.quote(self.a.peer_root)} && env {sets} timeout 1800 {' '.join(map(shlex.quote, argv))}"

    def drop(self, peer):
        if not self.a.drop_caches:
            return None
        rc, _ = sh(self.a.drop_cmd)
        if peer and not self.a.loopback:
            rc2, _ = sh(["ssh", "-o", "BatchMode=yes", self.a.peer_ssh, self.a.drop_cmd])
            rc = rc or rc2
        return rc == 0

    def share_secret(self):
        """One random key file on both hosts (mode 600), sent over ssh's stdin, never on a command line."""
        self.out.mkdir(parents=True, exist_ok=True)
        if not self.secret.exists():
            self.secret.write_bytes(os.urandom(48).hex().encode())
            self.secret.chmod(0o600)
        if self.a.loopback:
            return
        cmd = f"umask 077 && cat > {shlex.quote(self.a.peer_secret)}"
        with open(self.secret, "rb") as f:
            p = subprocess.run(["ssh", "-o", "BatchMode=yes", self.a.peer_ssh, cmd], stdin=f, capture_output=True)
        if p.returncode != 0:
            sys.exit(f"could not write the key on the peer: {p.stderr.decode().strip()}")

    def whole(self, set_, where="a"):
        """One engine on box A (here) or box B (the peer): its summary and its token bytes."""
        self.n += 1
        name = f"{self.n:02d}-whole-{where}-s{set_['streams']}-k{set_['k']}-d{set_.get('draft', 0)}"
        tok = self.out / f"{name}.tok"
        dropped = self.drop(where == "b")
        env = self.env(set_)
        if where == "a":
            p = subprocess.run(self.argv(self.a.bin, tok), env={**os.environ, **env}, capture_output=True,
                               text=True, timeout=1800)
            text, rc, data = p.stdout + p.stderr, p.returncode, tok.read_bytes() if tok.exists() else b""
        else:
            rtok = f"{self.a.peer_out}/{name}.tok"
            cmd = self.remote(env, self.argv(self.a.peer_bin, rtok)) + f" && cat {shlex.quote(rtok)} | base64 -w0"
            p = subprocess.run(["ssh", "-o", "BatchMode=yes", self.a.peer_ssh, cmd], capture_output=True, text=True,
                               timeout=1900)
            text, rc = p.stderr, p.returncode
            data = base64.b64decode(p.stdout.strip()) if rc == 0 and p.stdout.strip() else b""
            text = "\n".join([text, "\n".join(ln for ln in p.stdout.splitlines() if ln.startswith("pipe "))])
        (self.out / f"{name}.log").write_text(text)
        return self.record(name, set_, rc, text, data, dropped)

    def ring(self, set_):
        """Rank 1 on the peer (or here with --loopback), rank 0 here: its summary and its token bytes."""
        self.n += 1
        name = f"{self.n:02d}-ring-s{set_['streams']}-k{set_['k']}-d{set_.get('draft', 0)}"
        tok = self.out / f"{name}.tok"
        dropped = self.drop(True)
        env = self.env(set_)
        follow_log = open(self.out / f"{name}.r1.log", "w")
        if self.a.loopback:
            env1 = {**os.environ, **env, **self.ring_env(1)}
            follower = subprocess.Popen(self.argv(self.a.bin, os.devnull), env=env1, stdout=follow_log,
                                        stderr=subprocess.STDOUT)
        else:
            cmd = self.remote({**env, **self.ring_env(1)}, self.argv(self.a.peer_bin, "/dev/null"))
            follower = subprocess.Popen(["ssh", "-o", "BatchMode=yes", self.a.peer_ssh, cmd], stdout=follow_log,
                                        stderr=subprocess.STDOUT)
        try:
            p = subprocess.run(self.argv(self.a.bin, tok), env={**os.environ, **env, **self.ring_env(0)},
                               capture_output=True, text=True, timeout=1800)
            rc0, text = p.returncode, p.stdout + p.stderr
        finally:
            try:
                rc1 = follower.wait(timeout=120)
            except subprocess.TimeoutExpired:
                follower.terminate()
                rc1 = follower.wait(timeout=30)
            follow_log.close()
        (self.out / f"{name}.r0.log").write_text(text)
        rec = self.record(name, set_, rc0, text, tok.read_bytes() if tok.exists() else b"", dropped)
        rec["follower_rc"] = rc1
        rec["ok"] = rec["ok"] and rc1 == 0
        return rec

    def record(self, name, set_, rc, text, data, dropped):
        summary = parse_pipe(text)
        rec = {"name": name, **set_, "rc": rc, "ok": rc == 0 and summary is not None, "summary": summary,
               "tokens_sha256": hashlib.sha256(data).hexdigest(), "dropped_cache": dropped, "_data": data}
        self.log.append({k: v for k, v in rec.items() if k != "_data"})
        state = f"{summary['tok_s']:.2f} tok/s" if summary else f"FAILED rc={rc}"
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {state}", flush=True)
        return rec


def checks(r, a):
    """Tokens equal to one engine at k = 1 and k = 2, groups and solo equal to one batch, drafted equal to plain."""
    t = a.check_tokens
    base = {"streams": a.check_streams, "tokens": t}
    ref = r.whole({**base, "k": 0})
    out = {"reference": ref["name"]}
    for k in (1, 2):
        got = r.ring({**base, "k": k})
        out[f"two_boxes_k{k}"] = {"ok": got["ok"], **compare(ref["_data"], got["_data"], a.check_streams, t)}
    grouped = r.whole({**base, "k": 2})
    out["one_box_k2_groups"] = {"ok": grouped["ok"], **compare(ref["_data"], grouped["_data"], a.check_streams, t)}
    solo = r.whole({"streams": 1, "tokens": t, "k": 0})
    out["solo_equals_concurrent"] = {"ok": solo["ok"], "equal": prefix_equal(solo["_data"], ref["_data"], t)}
    if a.draft:
        d = r.whole({**base, "k": 0, "draft": a.draft})
        out["drafted_one_box"] = {"ok": d["ok"], "kept_drafts": (d["summary"] or {}).get("kept_drafts"),
                                  **compare(ref["_data"], d["_data"], a.check_streams, t)}
        d2 = r.ring({**base, "k": 2, "draft": a.draft})
        out["drafted_two_boxes_k2"] = {"ok": d2["ok"], "kept_drafts": (d2["summary"] or {}).get("kept_drafts"),
                                       **compare(ref["_data"], d2["_data"], a.check_streams, t)}
    out["pass"] = all(v.get("ok") and v.get("equal") for k, v in out.items() if isinstance(v, dict))
    return out


def cells(r, a):
    """Total tok/s at each stream count: each box alone, two boxes at k = 1 and k = 2, medians of `--repeats`."""
    rows = []
    for s in a.streams:
        runs = {k: [] for k in ("whole_a", "whole_b", "ring_k1", "ring_k2")}
        same = True
        for rep in range(a.repeats):
            arms = [("whole_a", lambda: r.whole({"streams": s, "tokens": a.tokens, "k": 0}))]
            if not a.loopback:
                arms.append(("whole_b", lambda: r.whole({"streams": s, "tokens": a.tokens, "k": 0}, "b")))
            arms += [("ring_k1", lambda: r.ring({"streams": s, "tokens": a.tokens, "k": 1})),
                     ("ring_k2", lambda: r.ring({"streams": s, "tokens": a.tokens, "k": 2}))]
            if rep % 2:
                arms.reverse()
            got = {}
            for key, fn in arms:
                got[key] = fn()
                if got[key]["ok"]:
                    runs[key].append(got[key]["summary"]["tok_s"])
            ref = got["whole_a"]["_data"]
            same = same and all(g["_data"] == ref for g in got.values() if g["ok"])
        row = {"streams": s, "tokens_equal_across_arms": same}
        for key, vals in runs.items():
            row[key] = statistics.median(vals) if vals else None
            row[key + "_runs"] = vals
        rows.append(row)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, epilog="Run on rank 0's host; rank 1 runs on the peer over ssh "
                                "(--peer-ssh) or here (--loopback). Writes <out>/receipt.json and <out>/receipt.md.")
    p.add_argument("--model", required=True, help="checkpoint dir (the same path on both hosts)")
    p.add_argument("--bin", required=True, help="tf-qwen35-cuda on this host")
    p.add_argument("--repo", default=".", help="this host's TensorFold checkout (its commit is recorded)")
    p.add_argument("--out", required=True)
    p.add_argument("--addr", required=True, help="rank 0's address on the link (this host)")
    p.add_argument("--peer-addr", help="rank 1's address on the link")
    p.add_argument("--peer-ssh", help="user@host for ssh to the peer")
    p.add_argument("--peer-root", help="the peer's TensorFold checkout (default: --repo)")
    p.add_argument("--peer-bin", help="the peer's tf-qwen35-cuda (default: --bin)")
    p.add_argument("--peer-out", default="/tmp", help="a scratch directory on the peer")
    p.add_argument("--loopback", action="store_true", help="rank 1 on this host too: overhead only, no speed claim")
    p.add_argument("--port", type=int, default=29600)
    p.add_argument("--layout", default="0:32,32:64")
    p.add_argument("--kv", choices=("bf16", "fp8"), default="bf16")
    p.add_argument("--prompt", type=int, default=512)
    p.add_argument("--tokens", type=int, default=128, help="decoded tokens a stream in the tok/s cells")
    p.add_argument("--streams", type=lambda v: [int(x) for x in v.split(",")], default=[4, 8, 16])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--check-streams", type=int, default=4)
    p.add_argument("--check-tokens", type=int, default=64)
    p.add_argument("--draft", type=int, default=3, help="prompt-lookup drafts in the drafted check (0: skip it)")
    p.add_argument("--drop-caches", action="store_true", help="drop the page cache on both hosts before every start")
    p.add_argument("--drop-cmd", default=f"sudo -n sh -c {shlex.quote(DROP)}", help="how (a shell command)")
    p.add_argument("--skip-cells", action="store_true")
    a = p.parse_args(argv)
    if a.loopback:
        a.peer_addr = a.peer_addr or a.addr
    elif not (a.peer_ssh and a.peer_addr):
        p.error("two boxes need --peer-ssh and --peer-addr (or --loopback)")
    a.peer_root = a.peer_root or str(Path(a.repo).resolve())
    a.peer_bin = a.peer_bin or str(Path(a.bin).resolve())
    a.peer_secret = f"{a.peer_out}/tf-pipe-secret"
    r = Runner(a)
    r.share_secret()
    hosts = {"a": facts(a.bin, a.repo, a.addr)}
    if not a.loopback:
        script = f"{a.peer_root}/tools/pipeline/pipe_receipt.py"
        cmd = f"python3 {shlex.quote(script)} --facts --bin {shlex.quote(a.peer_bin)} --repo {shlex.quote(a.peer_root)}"
        rc, txt = sh(["ssh", "-o", "BatchMode=yes", a.peer_ssh, cmd + f" --addr {shlex.quote(a.peer_addr)}"], 120)
        hosts["b"] = json.loads(txt) if rc == 0 else {"error": txt}
    rc, ping = sh(["ping", "-c", "20", "-i", "0.2", "-q", a.peer_addr], 60)
    m = re.search(r"= ([0-9.]+)/([0-9.]+)/([0-9.]+)", ping)
    link = {"layout": a.layout, "kv": a.kv, "loopback": a.loopback, "rtt_ms_avg": float(m.group(2)) if m else None}
    receipt = {"rfc": "ashhart/TensorFold#2005 pipelined rounds", "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
               "hosts": hosts, "link": link, "args": {k: v for k, v in vars(a).items()}}
    receipt["checks"] = checks(r, a)
    receipt["cells"] = [] if a.skip_cells else cells(r, a)
    receipt["runs"] = r.log
    (r.out / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str))
    md = [f"# Pipelined rounds receipt ({receipt['date']})", "",
          f"Link: {json.dumps(hosts['a'].get('link'))}, rtt {link['rtt_ms_avg']} ms; layout {a.layout}, kv {a.kv}.",
          f"Token checks: {'PASS' if receipt['checks']['pass'] else 'FAIL'} ({verdicts(receipt['checks'])})",
          "", table(receipt["cells"], "two stages, one box" if a.loopback else "two boxes")
          if receipt["cells"] else "(cells skipped)", ""]
    (r.out / "receipt.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0 if receipt["checks"]["pass"] else 1


if __name__ == "__main__":
    if "--facts" in sys.argv:
        f = argparse.ArgumentParser()
        f.add_argument("--facts", action="store_true")
        f.add_argument("--bin", required=True)
        f.add_argument("--repo", required=True)
        f.add_argument("--addr")
        x = f.parse_args()
        print(json.dumps(facts(x.bin, x.repo, x.addr)))
        sys.exit(0)
    sys.exit(main())
