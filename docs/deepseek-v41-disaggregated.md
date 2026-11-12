# DeepSeek-V4.1 disaggregated prefill and decode (GB10 pair + Apple Silicon)

A pair of NVIDIA GB10 machines (two DGX Spark or ASUS Ascent GX10, tensor-parallel over CUDA) **prefill** the prompt; an
Apple Silicon Mac (Metal, EXL3) **decodes** it and serves the API. The prefill state crosses on a ConnectX-5 cable through
[MCDMA](https://github.com/ashhart/MCDMA) (RDMA from macOS), or over plain HTTP where there is no such cable.

This page is the operator recipe. The wire format, the library and the exactness contract are in
[deepseek-v41-handoff.md](deepseek-v41-handoff.md); the model lanes are in [deepseek-v41.md](deepseek-v41.md).

> Status: the path works end to end (link, hand-off, landing, fallback) but, with the engines as they are, it does **not**
> lower time to first token: see [Measured](#11-measured). Use it for capacity, or when the pair prefills faster than the Mac.

> A hand-off prefix is outside the exactness contract: it is opt-in (`--prefill-from`), labelled `remote_prefix` in the
> reply, and decode after it is not bit-identical to a Mac-only prefill. Everything below keeps the Mac able to prefill
> by itself: a producer that is down, slow or wrong only costs the Mac its normal prefill.

## What runs where

```
   client ──HTTP──▶  Mac (Metal): `tensorfold-native serve ... --prefill-from http://PRODUCER:PORT --mcdma-link NAME`
                       │  1. trigger the export     POST /v1/completions + kv_transfer_params   (HTTP, LAN)
                       │  2. pull the state         MKVH protocol 2 over the link NAME          (RDMA, CX5 cable)
                       │  3. land it, run the last prompt token, decode
                       ▼
   GB10 pair (CUDA, TP=2): rank 0 `tensorfold-dsv41 serve ... --mcdma-produce NAME`, rank 1 `tf-dsv41 follow`
                       prefills the prompt to its last-but-one token (all but the Mac's tail), exports rank 0's state, holds it until pulled
```

Only the payload (about 3.7 KB per prompt token, 31 MiB for 8,704 tokens) uses the cable. The link's set-up datagrams and
the trigger use the LAN, so the Mac needs both.

## Requirements

| part | requirement |
|---|---|
| Mac | Apple Silicon with enough memory for the EXL3 pack (the 2.9 bpw pack needs about 215 GB resident), macOS 27, `rdma_ctl enable`, the MCDMA native driver installed, a ConnectX-5 in a Thunderbolt 5 enclosure |
| GB10 pair | two machines with the CUDA stack and the inter-machine RoCE link of the TP pair (this guide calls its devices the *pair devices*); one more ConnectX-7 port free for the Mac, and a free RDMA GID |
| cables | 100G passive QSFP28 whose EEPROM the ConnectX-7 accepts (see [Cables](#cables)) |
| LAN | both ends on one layer-2 segment (the exchange uses link-local datagrams with hop limit 255) |
| model | the same checkpoint files on both sides: the manifest names the checkpoint by the hash of `config.json` and `model.safetensors.index.json` |

## 1. The Mac

Follow MCDMA's `docs/install.md` (Recovery: reduced security, user-managed kernel extensions, `rdma_ctl enable`; install
the kext with the enclosure disconnected, approve it, reboot with the enclosure attached). Then:

```
kmutil showloaded --list-only | grep org.mcdma.cx5.native      # loaded
ibv_devices                                                    # rdma_mcrdma0, rdma_mcrdma1
ibv_devinfo -d rdma_mcrdma1 | grep state                       # PORT_ACTIVE once a cable links
```

## 2. Cables

The ConnectX-7 identifies a QSFP module from byte 0 of its EEPROM (`ethtool -m IF`: identifier `0x11`). A cable whose
EEPROM does not name a QSFP28 leaves the port without carrier: `mstlink -d PCI` reports status opcode 1036 ("Module's PMD
type is not enabled") and the port never links at any speed. Use a cable the card recognises (MCDMA's guide tested the
Mellanox MCP1600-C001E30N; the Spark user guide lists others). Check the port before going further:

```
ethtool IF | grep -E 'Link detected|Speed'                     # yes, 100000Mb/s
sudo mstlink -d PCI | grep -E 'State|Opcode'                   # Active, opcode 0
```

The two Mac ports may be crossed at the enclosure: if a byte-verified transfer fails with `CQ status=12` (retry exceeded),
swap which Mac interface you pair with which machine (step 4 finds the right one).

## 3. Each pair machine: the Mac-facing port

Leave the pair's own RoCE port and its settings alone. On the *other* port (here `PEER_IF`, RDMA device `PEER_DEV`):

```
sudo nmcli dev set PEER_IF managed no          # until reboot; stops NetworkManager replacing the MTU and addresses
sudo ip link set dev PEER_IF mtu 9000 up
cat /sys/class/net/PEER_IF/address             # PEER_MAC
for i in 0 1 2 3; do echo $i $(cat /sys/class/infiniband/PEER_DEV/ports/1/gids/$i) \
                          $(cat /sys/class/infiniband/PEER_DEV/ports/1/gid_attrs/types/$i); done
```

Use the RoCE v2 GID that is the MAC-derived link-local address (`fe80::` + the EUI-64 of `PEER_MAC`); its index is
`PEER_GID_INDEX`.

**Pin NCCL to the pair's own devices on every launch of the TP pair** (`NCCL_IB_HCA==PAIR_DEV_A,PAIR_DEV_B`, the exact-match
form). Once the Mac-facing port has carrier NCCL's autodetection adds it to the TP rings; a ring step then goes to the
wrong machine, and the pair hangs at start-up at 100 % CPU with `IBV_WC_RETRY_EXC_ERR` in NCCL's log.

## 4. Address the link and prove it

On each Mac interface that has a cable (`mcrdmaN`, MAC `MAC_HWADDR`) and its peer (`PEER_GID`, `PEER_MAC`):

```
# Mac, from the MCDMA checkout
sudo python3 tools/restore-rdma.py --interface mcrdmaN --expected-mac MAC_HWADDR --peer-gid PEER_GID --peer-mac PEER_MAC
# peer: the reciprocal static neighbour (the Mac's address-only interface answers no neighbour discovery)
sudo ip -6 neigh replace MAC_GID lladdr MAC_HWADDR nud permanent dev PEER_IF
```

`MAC_GID` is the Mac's MAC-derived link-local address. Both entries are lost on reboot: put them in a boot script on each
side. Then run MCDMA's four-way byte check (`tools/native_cross_host.py`, see its `docs/install.md` step 5) for each pairing
and keep the pairing that passes; require WRITE and READ in both directions, bytes verified, in all three modes.

## 5. The link daemons

`mcdma-rpcd` carries the request/reply mailbox. Its set-up exchange runs on a shared **LAN** segment, not on the cable: the
`VIA` of each end is `LAN_IF/fe80::OTHER_ENDS_LAN_LINK_LOCAL`. (Naming the cable's interface fails with
`connect failed with status 6`.)

```
# producer machine (rank 0's host), as the user that runs the server, started BEFORE the server
mcdma-rpcd listen NAME PEER_DEV PEER_GID_INDEX 4096 LAN_IF/fe80::MAC_LAN_LINK_LOCAL:18620 4 64

# Mac, in a session that holds the "Local Network" permission (a Terminal or an ssh session; a detached process gets
# HostUnreachable to LAN hosts)
mcdma-rpcd connect NAME,en0/fe80::PRODUCER_LAN_LINK_LOCAL,18620,rdma_mcrdmaN,0,4096,4,64
```

The listen daemon logs `peer ready over roce` once the Mac connects. `NAME` is the link name used below.

## 6. The prefill pair

Rank 1 is `tf-dsv41 follow`; rank 0 is `tensorfold-dsv41 serve` with one more flag. The pair needs a profile whose
prefill state can be exported: replicated KV, whole-prompt prefill, a paged pool. It may run
`TF_DSV41_PREFILL=replay` (the cut stands after the decoder replay; see below). It must **not** run
`TF_DSV41_PREFILL_PIECES=1` (a prompt in pieces never stands at the cut) or `TF_DSV41_KV_SPLIT=1` (half the pages are on
the other rank); the server refuses to start, by name, with either.

```
# both machines
export NCCL_IB_HCA==PAIR_DEV_A,PAIR_DEV_B TF_DSV41_FORMS_INPLACE=1 TF_DSV41_SAMPLING=1 TF_DSV41_POOL_TOKENS=65536
export TF_DSV41_GM_V2=1 TF_DSV41_GM_V3=1 TF_DSV41_PF_DENSE=fused TF_DSV41_PF_DENSE_TABLE=$TABLE TF_DSV41_PF_OVERLAP=4
export TF_DSV41_L2PF=1 TF_DSV41_L2PF_MB=12 TF_DSV41_L2PF_PACE_GBPS=150 TF_DSV41_PREFETCH_AHEAD=1 TF_DSV41_ENGRAM_AIO=1
export TF_DSV41_BRANCHES=1 TF_DSV41_BRANCHES_PRIO=side
# rank 0 only
tensorfold-dsv41 serve $MODEL --host 0.0.0.0 --port 8000 --name DeepSeek-V4.1-Flash-TF --no-thinking \
    --mcdma-produce NAME --api-key-file $KEYFILE
```

Do **not** set `TF_DSV41_HANDOFF_DUMP` in service: it writes each export to disk on the engine thread.

Rank 0 logs `mcdma handoff: attached to the listen daemon on link NAME` and `exporting DeepSeek-V4.1 states as MKVH protocol 2
on link NAME`; a start refusal names the knob.

### How fast the pair prefills, and why not faster

The export-compatible profile above is **not** the pair's fastest. Measured on two GB10 (TP=2), tokens per second, best of
three, prompt length 1k / 8k / 32k:

| profile | 1k | 8k | 32k | exportable |
|---|---|---|---|---|
| serving profile (replay prefill, 4,096-row pieces, split KV) | 1.60k | 2.16k | 2.53k | no |
| replay prefill, 4,096 rows, replicated KV, no pieces | 1.64k | 2.17k | 2.59k | yes |
| export profile, no tuning | 0.98k | 1.18k | 1.19k | yes |
| export profile with the knobs above | 1.04k | 1.19k | 1.23k | yes, byte-identical to the untuned export |

The knobs above add 0.5 to 6 %: they are optional. What makes the serving profile fast is replay prefill
(`TF_DSV41_PF_4K`, `TF_DSV41_PF_TBO` and `TF_DSV41_PREFILL_ROWS` above 2048 are refused without it). Replay computes the
decoder layers over the last 127 rows only, at the cut: the export carries those layers' rings from that replay (the
window's oldest row is zero), so the decoder layers' rings differ from a full prefill's (normalized RMS 0.3 to 0.5, growing
with layer depth) while the sliding-window rows below layer 20, the carries, the compressed pages and the index keys are
byte-identical. Prefill on the plain export profile costs about 0.83 ms a row at every length from 1k to 32k: it is GPU-compute bound, not Engram bound (7 %).
A prompt of 61k tokens (the pool's 65,536-token cap) exports in 51 s.
Of the knobs tried, `TF_DSV41_ENGRAM_GATE`, `TF_DSV41_ENGRAM_PREFETCH`, `TF_DSV41_INDEX_BUDGET_MIB`,
`TF_DSV41_ROUTER_GROUP_ROT` and `TF_DSV41_KV_NORM_STORE` together made the export fail (HTTP 409, "the prompt pass ended
before its state was saved") and bought nothing; which one is the cause was not bisected.

The end-to-end comparison against a Mac that prefills at about 0.9 to 1.0k tokens a second is in [Measured](#11-measured).
Freeing the Mac's GPU from long prompts while it decodes other requests is the other expected gain; it is not measured here.

## 7. The Mac decoder

```
TF_PREFILL_API_KEY_FILE=$KEYFILE tensorfold-native serve $PACK --engram-stream-dir $ENGRAM \
    --chat-template $PACK/chat_template.jinja --context 32768 --prompt-cache-gib 4 \
    --prefill-from http://PRODUCER:8000 --mcdma-link NAME
```

`--prefill-from` is the producer's HTTP address (the trigger); `--mcdma-link NAME` is the daemon link (the payload).
`TF_DSV41_PREFILL_STEP=4096` is the Mac's own prefill step. A prompt shorter than `TF_PREFILL_MIN` is prefilled on the
Mac.

The Mac runs only the prompt's last token after the pull (`TF_PREFILL_TAIL`, default 1: a one-row, decode-class step, so no
prefill pass runs on the Mac), and the pair prefills all the rest, tail included. `TF_PREFILL_TAIL=0` restores the 512-token
boundary (`TF_PREFILL_CHUNK`), which leaves the Mac a 96 to 461 token pass of 0.6 to 1.1 s. Not measured on the pair yet; see
[the hand-off page](deepseek-v41-handoff.md#how-much-of-the-prompt-is-handed-over) for what moves, what is not shipped and the drafter's start.

Leave `--keep-warm` at its default (900 s) and do not pass `--keep-warm 0`: while the pair prefills, the Mac's GPU is idle for
seconds, macOS un-wires an idle model after one to two seconds, and the first pass after the hand-off then waits 0.9 to 2.3 s
to re-wire it. The keepalive ticks while a hand-off is awaited. A quiet server (a request after more than a few seconds of
idleness) can still pay that re-wire once.

### Several requests, and when a prompt goes to the pair at all

The pair prefills one prompt at a time, and the Mac holds one hand-off at a time. `TF_PREFILL_ROUTE=auto` (the default)
decides each prompt that reaches `TF_PREFILL_MIN` when it is queued, and again while it waits for a lane:

* **The pair is busy with another prompt**: the prompt is prefilled on the Mac at once, as a short prompt is, unless the pull
  under way ends so soon that waiting for it and then pulling still beats the Mac's own prefill. It is not queued behind the
  pull with the Mac idle (a mutex held across the whole pull did that).
* **A stream is decoding on the Mac**: the prompt goes to the pair even if the pair is no faster, because a prompt pass on
  the Mac would stall that stream for seconds (inter-token latency isolation), unless the pair is busy as above.
* **Otherwise** it goes to the pair only if the pull is predicted no slower than the Mac's own prefill: the prefix at the
  pair's measured rate plus the Mac's last-token pass, against the rest of the prompt at the Mac's measured rate. The rates
  are running averages of the pulls and prompt passes this server timed (a pass under 1024 rows says too little to count;
  one sample moves a rate by a third to three times at most). Until it has timed its own it assumes 950 tokens a second for
  the Mac (`TF_PREFILL_LOCAL_TPS`), 1000 for a pull (`TF_PREFILL_REMOTE_TPS`) and 1 s for the last-token pass, the export
  profile's numbers: with them a lone prompt of under about 19k tokens stays on the Mac, whose prefill is as fast as the
  pair's there (the pair's gain is the Mac's freedom, not the first token). The pair's rate is learned only from pulls, so
  a pair that is faster than that (a replay export) is first used when a stream decodes or the prompt is long; set
  `TF_PREFILL_REMOTE_TPS` to what it does to use it from the start.

The server logs `remote prefill: a prompt of N tokens (M cached) is prefilled here: REASON` once for each reason a prompt
is kept off the pair for now. `TF_PREFILL_ROUTE=always` restores the old behaviour: every prompt past `TF_PREFILL_MIN` goes
to the pair, queued behind any pull under way, however long that is. Not yet measured on the machines: two concurrent 8k
prompts, and the rates the average settles at.

### A pair that goes away

* **Deadline.** A pull is bound by `TF_PREFILL_TIMEOUT_S` if set, else by 15 s plus three times the pull the Mac's own
  timings predict for it (the prefix at the pair's measured rate plus the last-token pass), between 30 s and 30 min; the
  trigger and, over HTTP, the transfer each get that. A pair that stopped answering is found in about a minute, not after
  an hour-long default.
* **Circuit breaker.** After `TF_PREFILL_BREAKER` failed pulls in a row (default 3; the link down, a connect or trigger
  that timed out; a refusal or a mismatch is an answer and does not count) the breaker opens: prompts are prefilled on the
  Mac (the log says why once: `the pair is not answering`), and nothing is sent to the pair for `TF_PREFILL_COOLDOWN_S` (default
  30). Then one prompt is the trial: `GET /health` on the GB10 server (2 s) and, over MCDMA, a re-open of the link, then the
  pull. If it succeeds the breaker closes (`remote prefill: the pair answers again`); if not it opens for another
  cool-down. `TF_PREFILL_BREAKER=0` never stops pulling.
* **The connect daemon restarts.** `mcdma-rpcd connect` unlinks and recreates its mailbox at every start, and a mapping
  of the old object never sees the new link. Before each pull the server checks that the mailbox name still maps the
  object it holds (POSIX shared memory on a Mac has no inode, so it writes a nonce into the payload bytes of its own
  mapping, which nothing reads between calls, and looks for it through a new mapping of the name) and, if the daemon
  replaced it, maps the new one (backoff 1 s to 30 s while that is refused) and makes a new client: no server restart
  (`remote prefill: the connect daemon's mailbox was replaced; the link was re-opened`). The producer's listen daemon
  restarting is healed on the GB10 side (the producer attaches again, at most once a second).

### A request that goes away

The server tells a pull when its request ended (the client hung up, or the request was cancelled): a pull waiting at the
gate, for the trigger's reply, or between the decoder's calls stops at once and the export it opened is released, and the
Mac does not log a failure. The GB10's prompt pass itself is **not** stopped. The two ranks run it in lock step (the
follower is driven by the leader's every pass), so abandoning it on one rank would leave the other waiting on a
collective: it runs to its end, the export it makes is dropped by the producer when its time to live (120 s) ends, and a
hand-off request cancelled before its cut is not captured at all.

## 8. Verify

A reply with a long prompt carries, in its `tensorfold` block, `"exact_mode": "remote_prefix"`, `remote_prefix` and
`pulled_prefix` (tokens), plus `remote_route` and the pull's phases (`remote_queue_seconds`, `remote_trigger_seconds`, `remote_pull_seconds`,
`remote_land_seconds`, `remote_tail_seconds`, `remote_bytes`, `remote_tokens_per_second`; see `docs/api.md`), and the server logs one
`remote prefill: N tokens pulled: gate .., trigger .., transfer ..` line for each. A failed pull logs `remote prefill failed (REASON); the prompt is prefilled here` and the reply has
none of those fields. To prove the bytes used the cable, not the LAN, read the producer port's counters around a request:

```
sudo ethtool -S PEER_IF | grep tx_bytes_phy      # grows by about 3.7 KB per prompt token
```

## 9. Running it as services

Start order matters: the producer's listen daemon, then rank 1, then rank 0 (which attaches to the daemon while it
starts), then, at any time, the Mac's connect daemon and its server. Stop in the reverse order.

On each pair machine, a boot unit restores what a reboot loses (the static neighbour and the unmanaged port; the Mac's MAC and
link-local address are fixed for its CX5):

```
# /usr/local/bin/mac-link-up  (run by a oneshot unit After=network-online.target)
nmcli dev set PEER_IF managed no
ip link set dev PEER_IF mtu 9000 up
ip -6 neigh replace MAC_GID lladdr MAC_HWADDR nud permanent dev PEER_IF
```

```
# mcdma-rpcd-listen.service on the producer machine, User=the serving user, LimitMEMLOCK=infinity
ExecStart=/path/to/mcdma-rpcd listen NAME PEER_DEV PEER_GID_INDEX 4096 LAN_IF/fe80::MAC_LAN_LINK_LOCAL:18620 4 64
# the rank-0 unit: Requires=mcdma-rpcd-listen.service After=mcdma-rpcd-listen.service, and the NCCL_IB_HCA pin in its Environment=
```

On the Mac the daemon, then the server, run in the user's login session (a launchd *agent*, not a system daemon, so that
macOS Local Network access applies to the logged-in user). The Mac server falls back to its own prefill whenever the producer
is down, so the units do not depend on each other.

The Mac's restore command (`tools/restore-rdma.py`) and the peer's neighbour are lost on reboot of either side: run both at boot.


## 10. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| port has no carrier; `mstlink` opcode 1036 | the cable's EEPROM names no QSFP28 (generic cables) | a cable the card recognises |
| `CQ status=12` in the byte check | cables crossed at the enclosure, or no static neighbour | pair the other Mac interface; install both neighbours |
| `connect failed with status 6` | the exchange was given the cable's interface | give the LAN interface and the pinned peer address |
| the TP pair hangs at start at 100 % CPU, NCCL `RETRY_EXC_ERR` | NCCL autodetected the Mac-facing port | `NCCL_IB_HCA==PAIR_DEV_A,PAIR_DEV_B` |
| rank 0 exits: `no mcdma-rpcd listen daemon serves that link` | the daemon was not running when rank 0 reached the exporter | start the listen daemon first, as the same user |
| `HostUnreachable` from the Mac's server or daemon | macOS Local Network permission for a detached process | run both in a Terminal or ssh session |
| `link down (the connect end is silent)` in the listen log | the Mac daemon restarted or slept; it reconnects | none, unless it stays down |
| reply has no `remote_prefix` | prompt under `TF_PREFILL_MIN`, producer down, checkpoint hash differs | read the Mac's `remote prefill failed (...)` line |
| `pkill -f PATTERN` ends your ssh command | the pattern matches the command line carrying it | kill by pid |
| the first pass after a hand-off takes 1 to 2 s longer than the tail's size explains | the idle model was un-wired while the pair prefilled | keep `--keep-warm` at its default; not `0` |

## 11. Measured

Two GX10 (GB10, TP=2) and an M5 Ultra Mac Studio (256 GB), the 2.9 bpw EXL3 pack, 100G cables, MCDMA 0.1.18, macOS 27.0:

| item | value |
|---|---|
| link | 100 Gb/s; MCDMA four-way byte checks pass in all three modes |
| round trip, 4 KiB | 17 µs median (`fabric-check`, host memory, one queue pair) |
| streaming, one queue pair, one direction | 10.8 Gbit/s; 28 MiB mailbox reply 17.3 Gbit/s (the Thunderbolt-5 PCIe tunnel is the ceiling: MCDMA documents about 29 Gbit/s writes, 50 Gbit/s reads) |
| hand-off size | 31 MiB for 8,704 tokens (about 3.7 KB a token); 3 requests, all `remote_prefix`, correct answers |
| bytes on the cable | 34.2 MB sent by the producer's port in one request; the LAN carried about 140 KB |

### End to end: time to first token, hand-off against the Mac alone

Same hardware, 2.9 bpw pack, the tuned prefill profile, three prompts per size (distinct text), seconds, median of three
(`max_tokens` 1), Mac alone on the same binary and prompts. The Mac's tail is the part of the prompt past the last
512-token cache boundary of the prefix (96 to 461 tokens). Measured twice: before and after two fixes on the Mac
(the idle model is kept wired while a hand-off is awaited, and the pass is planned from the landed prefix instead of from token 0).

| prompt tokens | prefix handed over | hand-off, before fixes | hand-off, after fixes | Mac alone | after / Mac alone |
|---|---|---|---|---|---|
| 1k | none (below `TF_PREFILL_MIN`) | 1.62 | | 1.62 | 1.00 |
| 4k | 4,096 | 7.11 | 4.93 | 4.61 | 1.07 |
| 8k | 7,680 to 8,192 | 11.89 | 8.88 | 8.34 | 1.06 |
| 16k | 15,872 to 16,384 | 17.90 | 16.10 | 15.78 | 1.02 |
| 30k | 29,184 to 30,208 | 32.17 | 29.12 | 28.96 | 1.01 |

**The hand-off does not shorten time to first token on this hardware.** After the fixes it is level with the Mac alone (within 1 to 7 %):
the pair prefills and exports at about 1.0 to 1.1k tokens a second, which is what the Mac does alone (0.9 to 1.07k), and the pair's
time is 93 to 95 % of the hand-off's. The Mac's landing plus tail is 0.6 to 1.1 s (it follows the tail's length; it was 1.7 to 4.0 s before
the fixes), and the pull adds 0.1 to 0.55 s. A hand-off wins only when the pair prefills substantially faster than the Mac:
the serving profile reaches 2.2 to 2.5k tokens a second but cannot be exported (see above).

What is checked and works: the payload crosses the cable (the producer port's `tx_bytes_phy` grew 3.7 to 3.9 KB per exported
token while the LAN carried about 68 KB); the Mac falls back to its own prefill when the link is down
(`remote prefill: the link is not usable (LinkDown); prompts are prefilled here`, `exact_mode: exact`); a quiet server (12 s and
30 s idle) shows no wake-up penalty; a second turn of a conversation asks the producer again for the same prefix (the producer
answers from its pool in 0.1 s) and the Mac answers in 1.0 s of TTFT. Greedy decoding after a hand-off differs from a Mac-only
prefill within 5 to 10 tokens on three 9 to 10k-token prompts, with fluent, on-topic text on both: that is the `remote_prefix`
contract, not a defect, and no quality benchmark has been run on it. Not measured: a producer HTTP failure mid-transfer,
concurrent requests, quality, `--mcdma-link http` on the final build.

The remaining case for it is capacity, not latency: a pair that is otherwise idle can take long prompts off the Mac's GPU (expected, not measured).
