# tf-mcdma-pair: the #1594 gate for a ConnectX link

The gate: on a Mac and a Linux host over ConnectX (RoCE v2), every collective equals a host reference on both ends, a
stopped peer or a dropped link fails by name rather than hanging, and a Thunderbolt link behaves as before.
`tf-mcdma-pair` (zig/tests/mcdma_pair.zig) runs one rank per process; the scripts here make the hardware window
"copy, run, collect" and prove the whole flow offline first. No production code depends on them.

## The binary

```
tf-mcdma-pair SETTINGS.json [overrides]                        one rank (the hardware runs)
tf-mcdma-pair --pair OUTDIR RANK0.json RANK1.json [overrides]  both ranks in one process (MCDMA's offline stub verbs)
```

`zig build tf-mcdma-pair` builds it for this host, and `zig build tf-mcdma-pair-dist` for aarch64-macos,
aarch64-linux-gnu and x86_64-linux-gnu (linked against glibc 2.35, needing its symbols up to GLIBC_2.34) under
zig-out/bin/mcdma-pair/<triple>/. `--help` lists the overrides and exit codes; the settings fields are `Settings` in
zig/tests/mcdma_pair_rank.zig.

It runs barrier, broadcast from both roots, bf16 and f32 all-reduce (against `collective.reduce` and an fp32 rank-order
reference), uneven all-to-all, gather from both roots, pipes both ways, a bulk copy across the Mac's 4 MiB registration
pieces, a signal storm that checks every landed value, and rounds of write2Signal and writeSignalFrom. Each test prints
PASS or FAIL with a SHA-256 over its result buffers in an order both ranks share, beside the same hash over the host
reference. The ranks first swap a run id and their settings and refuse a mismatch by name. A watchdog names a step that
stops making progress (exit 4).

Arms: `exit` (the victim closes its endpoint mid-run, MCDMA sends BYE, and it exits 0), `kill` (it SIGKILLs itself, no
BYE), `stop` (it posts nothing more but keeps its endpoint open), `hang` (it spins, and its own watchdog must name it),
and `external` (an operator drops the link mid-run). The survivor must fail by name: PeerDown, PeerTimeout or Timeout,
with MCDMA's reason. On a Mac, a ConnectX window whose window and both ranks' staging take more than 63 pieces of
4 MiB is refused by name (TooManyRegions) before anything opens: the kext registers at most 64 regions per context.

## The scripts

- `run_macos.sh` and `run_linux.sh` (sharing `common.sh`) run one side from a settings JSON and gather its log, exit
  code, settings and read-only host facts into results/<stamp>-<label>-rank<N>-<host>/ (build/mcdma-pair/results in
  the repository). Nothing in them uses sudo. The pre-flight WARNs, and meta.txt records:
  - on a Mac: the mcrdma port not UP and RUNNING, MCDMAPortActive not Yes, no permanent ndp neighbour for the peer's GID
    at its MAC, relaxed ordering on, the loaded kext not the settings' `expected_kext_uuid`, the meeting interface not
    holding `meeting_addr`, the clock more than 1 s off (sntp only reads it), and the application firewall on;
  - on Linux: the port not ACTIVE, the gid not RoCE v2, active_mtu below the path MTU, memlock below the window plus
    staging, no permanent `ip -6 neigh` entry for the peer's GID at its MAC, the meeting interface missing or not
    holding `meeting_addr`, and the clock not NTP-synchronized or more than 1 s off.

  A Mac's ConnectX link needs the MCDMA kext and provider loaded, the cabled mcrdmaN port up with its RDMA state restored
  (restore-rdma.py, by the operator with sudo), device `rdma_mcrdmaN`, and gid 0 (the kext has only GID index 0). A Linux
  host (a GB10 with a ConnectX-7, or an x86_64 Linux host with a ConnectX-4 Lx) needs its port ACTIVE, the settings' gid
  a RoCE v2 entry, active_mtu at least the path MTU and the Mac's static neighbour. Environment: `TF_MCDMA_PAIR_BIN`,
  `TF_MCDMA_PAIR_RESULTS`, `TF_MCDMA_PAIR_OUT` (this run's directory), `TF_MCDMA_PAIR_OUTER_S` (a last-resort kill -9
  for a process its watchdog cannot reach; default 900).
- `build_mcdma.sh [--stub|--loop] [MCDMA_DIR [OUT_DIR]]` builds libmcdma-fabric as MCDMA's rpc/Makefile does (Apple's
  librdma on a Mac, rdma-core's libibverbs on Linux), or on MCDMA's stub verbs (`--stub`, as its tests/test_link.py
  builds it), or `loop_fabric.c` (`--loop`). MCDMA_DIR defaults to TF_MCDMA_SRC, else the bundle's mcdma/. It prints
  the path, its sha256 and what it links, and installs nothing.
- `settings.py HOSTS.json OUTDIR [--root DIR]` writes every run's settings and a PLAN.txt per host pair from a hosts
  file. `hosts.example.json` documents every field with documentation addresses; real hosts files stay outside the
  tree. A `label_prefix` such as `"r-"` keeps a rehearsal's runs (r-G1 ..) apart from the gate's (G1 ..). PLAN's lines
  copy the bundle, prepare each port (the operator's sudo lines, each with a read-back), run each host's side and compare.
- `bundle.sh [HOSTS.json ...]` (with `ZIG` and `TF_MCDMA_SRC` set, from the TensorFold root) packs the three binaries,
  these scripts, MCDMA's fabric sources and each pair's settings into build/mcdma-pair/tf-mcdma-pair-bundle.tgz. On
  each host, from the unpacked bundle: `./build_mcdma.sh`, optionally `./stub_proof.sh`, then PLAN's lines.
- `compare.py RESULTS... [--label L] [--kind cx5|tb5] [--settings DIR] [--same-as RESULTS... [--same-as-label L]]
  [--receipt FILE] [--within S] [--skew S] [--dropped-at T] [--min-rounds N] [--partial] [--same-host] [--stub]
  [--allow-warn]` gives the verdict, below.

## The runs

"a" is the Mac (the first host in the hosts file) and "b" the Linux host.

| run | what | the gate |
| --- | --- | --- |
| G1 | a rank 0, b rank 1, the whole suite | every result equals the host reference on both ends; equal hashes |
| G2 | the ranks swapped | the same |
| G3a | b SIGKILLs itself mid signal storm | a fails by name within 12 s |
| G3b | b closes its endpoint mid all-reduce | a fails by name with MCDMA's goodbye, or a completion error |
| G3d | b stops posting mid pipes, its endpoint open | a fails with PeerTimeout at 10 s, the link up |
| G5a | b alone at path MTU 4096 | OpenFailed, logged as DEVICE, only on an ACTIVE port whose active MTU is below 4096; else N/A |
| G5b | each host alone, gid -1 and kind cx5 | NeedGidIndex before anything opens |
| G5c | each host alone, gid -1 and no kind | OpenFailed with MCDMA's "<device>: gid -1", never "port not active" |
| G5d | a alone, no peer, connect_timeout_s 30 | ConnectFailed after about 30 s, logged as TIMEOUT |
| G4 | an operator drops b's link during a 300 s run | both fail by name soon after the recorded drop time; only for a host marked free |
| G3c | a SIGKILLs itself mid signal storm | b fails by name; last of all, since a killed Mac process may leave its port needing a restore |

PLAN ends with the Thunderbolt line: two Thunderbolt Macs' run of G1's suite (label tb-G1) must give the same per-test
hashes and suite settings as the pair's G1.

## compare.py's verdict

- A gate is two hosts over MCDMA: the ranks report different hosts (`--same-host` relaxes that for an offline proof),
  and each ran MCDMA's libmcdma-fabric linking the host's verbs, as meta.txt's links lines show (`--stub` accepts the
  offline stub or loop_fabric). A run that passes only because `--same-host` or `--stub` relaxed a check is OFFLINE PASS
  (not the gate), never GATE PASS.
- The pre-flight: every WARN is shown, and a wrong kext, relaxed ordering or a clock WARN fails the gate (notes with
  `--allow-warn`).
- Both logs are one run (the same RUN id), and each rank ran the plan's settings (`--settings DIR`, or the bundle's
  settings/*/<label>/), its EXPECT-LOG and REFUSE-LOG lines included.
- Every PASS hash equals its host reference on each rank, the whole suite ran every round (unless `--partial`), and the
  two ranks agree test by test and buffer by buffer, with the same run hash, link kind and path MTU.
- An arm's outcome, with MCDMA's reason: the victim did its part, and the survivor failed by name within `--within`
  (12 s) of its step, at or after the arm's test (in a lockstep test, its step too), within `--skew` (2 s) of the
  victim's ARM by the two wall clocks, and exited 3. G4: both ranks failed within the window around the drop time
  (`--dropped-at`, or the <label>.dropped_at file the operator wrote), after at least `--min-rounds` (3) rounds.
- A startup run: the named failure it expected, with its exit, and every EXPECT-LOG text present and no REFUSE-LOG text.
- `--same-as`: another run has the same per-test hashes and suite settings (a Thunderbolt run against a ConnectX run).

Exit codes: 0 GATE PASS, 1 FAIL, 2 bad input, 3 PARTIAL (`--partial`), 4 N/A (this host cannot show what the run is
for), 5 OFFLINE PASS. `--receipt FILE` writes the verdict with each rank's host facts and WARNs.

## The offline proof

`stub_proof.sh [OUTDIR]` runs the harness through the same run scripts and compare.py: over MCDMA's stub verbs (both
ranks in one process, nothing opened), over `loop_fabric.c` for the hardware window's two-process flow, and through
the runs compare.py must refuse. Every run is labelled stub-* or loop-*, OUTDIR defaults to proof/ (never results/),
and every compare passes `--same-host --stub --allow-warn`, so its best verdict is OFFLINE PASS. It checks a block of
loopback ports free first. The cases:

- ConnectX at path MTU 1024 and 4096 and Thunderbolt, with the same per-test hashes;
- the exit, stop and hang arms on both kinds, and mismatched settings refused at the handshake;
- the named startup failures: a port down, gid -1 on ConnectX, G5c's gid -1 with no kind, no peer, a path MTU MCDMA
  would not run, and on a Mac a window past the kext's 64 regions;
- settings.py's G1, G2, G3a, G3b, G3d, G4 and G3c in two processes over loop_fabric, G5c from its plan file, and PLAN's
  Thunderbolt line as written;
- the runs compare.py must refuse, each for its own reason, and the pre-flight WARNs no gate passes with.

`loop_fabric.c` is a stand-in libmcdma-fabric (ABI 1), not MCDMA: two processes on one host meet over a loopback TCP
socket, a write is a message the peer copies into its window, a closing peer sends BYE, and devices named tb* report
Thunderbolt, any other ConnectX (refusing the fused write-and-signal, as MCDMA does on one RoCE link). It speaks first
with a hello carrying `LOOP_FABRIC_TOKEN`, so it never talks to a stranger's socket. Its faults, for the must-fail cases
only:

| variable | fault |
| --- | --- |
| `LOOP_FAULT_NO_BYE=1` | disconnect sends no BYE |
| `LOOP_FAULT_SILENT_EOF=1` | a closed socket is not reported; only the next post fails |
| `LOOP_FAULT_CORRUPT_NTH=N` | flip a bit of the Nth received write of at least `LOOP_FAULT_CORRUPT_MINLEN` bytes |
| `LOOP_FAULT_SIGNAL_NTH=N` | corrupt the Nth received signal, left in place `LOOP_FAULT_SIGNAL_HOLD_MS` |
| `LOOP_FAULT_FLAP_MS=N` | shut the socket N ms after connect; `LOOP_FAULT_DROPPED_AT=FILE` records the time |
| `LOOP_FAULT_DROP_LEN=L` | lose the first write of exactly L bytes and the signal after it, the sender told OK |
