# Speed-up mode: one model on two Macs

Speed-up mode runs Qwen3.8 Flash Next 6-bit on two Macs joined by Thunderbolt. Each Mac holds the whole model
and does half the work of every request: prompts split their rows between the Macs, and each decode round splits
its DeltaNet heads, routed experts and vocabulary head. One Mac serves the HTTP API and the other runs every
request beside it. It needs the native Zig server (`tensorfold-native`) with the Flash Next replay engine, and
MCDMA for the Thunderbolt link. [The speed-up mode guide](../speed-up-mode.md) walks through the setup step by step,
with this build's speeds.

## What it gives

Measured on two M5 Ultra Macs (256 GB each) over one Thunderbolt 5 cable, greedy, 256-token replies, against
the same server build on one of the Macs:

| Prompt | Time to first token, two Macs / one | Decode, two Macs / one (tok/s) |
| --- | --- | --- |
| 1k code | 0.25 s / 0.34 s | 220 / 164 |
| 1k edit | 0.30 s / 0.37 s | 374 / 299 |
| 1k chat | 0.27 s / 0.34 s | 166 / 148 |
| 8k code | 1.27 s / 2.12 s | 189 / 157 |
| 8k edit | 1.22 s / 2.08 s | 356 / 290 |
| 8k chat | 1.12 s / 1.91 s | 189 / 144 |
| 32k code | 4.07 s / 7.08 s | 196 / 169 |
| 32k edit | 4.09 s / 7.10 s | 359 / 293 |
| 32k chat | 4.00 s / 6.89 s | 178 / 141 |

Prompts run 1.6 to 1.7 times as fast from 8k tokens (about 8,000 tokens a second at 32k). Decode gains less, 1.1
to 1.3 times: each layer still exchanges results between the Macs, and the draft head and the window's fixed
per-layer work run on both.

Both Macs produce the same reply token for token, at every draft depth. The experts give one Mac's bits: each Mac
runs every expert for half of a window's rows and they swap the results. A reply can still differ from one Mac's.
The two halves of the DeltaNet output projection are added in a different order, at the same fp32 precision.
Prompt splitting gives one Mac's bits exactly.

## What you need

- Two Apple silicon Macs, each with enough memory for the whole model. Flash Next 6-bit is 158 GB of weights, and
  each server takes 172 GB once loaded. Keep each server at or under 70% of its Mac's memory, 179 GiB on a 256 GB
  Mac. The server sizes its prompt cache to fit under that line, about 5 GiB on these Macs. A 192 GB Mac is too
  small. Put the same checkpoint and dump on both.
- A Thunderbolt 5 cable between them with RDMA enabled, and MCDMA's fabric library (`libmcdma-fabric.dylib`) built
  on each Mac from MCDMA's `main` branch, which carries its Thunderbolt links.
- Greedy decoding (temperature 0), as the Flash Next replay engine requires.

## Settings

Each Mac gets a small JSON file naming its rank, the MCDMA library and the link to the other Mac. On the Mac that
serves (rank 0):

```json
{"rank": 0, "library": "/path/to/libmcdma-fabric.dylib",
 "links": [{"peer": 1, "device": "rdma_en4", "via": "en4/192.0.2.2", "port": 7490, "name": "speedup"}]}
```

On the other Mac (rank 1), the same with `"rank": 1`, `"peer": 0` and the first Mac's address on that cable:

```json
{"rank": 1, "library": "/path/to/libmcdma-fabric.dylib",
 "links": [{"peer": 0, "device": "rdma_en4", "via": "en4/192.0.2.1", "port": 7490, "name": "speedup"}]}
```

`device` is the Thunderbolt RDMA device, `via` the interface and the other Mac's IPv4 address on that cable (an
IPv6 link-local address works too), and `port` a UDP port both ends reserve for meeting.

With two Thunderbolt cables between the Macs, MCDMA can bond them as one link (its dual-pipe build): join the two
devices and the two `via` entries with `+`, in the same order on both Macs, and keep the next port free as well:

```json
{"rank": 0, "library": "/path/to/libmcdma-fabric.dylib",
 "links": [{"peer": 1, "device": "rdma_en4+rdma_en3", "via": "en4/192.0.2.2+en3/192.0.2.6", "port": 7490, "name": "speedup"}]}
```

On two M5 Ultras the bond nearly doubles bulk throughput (96 against 53 Gbit/s); in speed-up mode it brings the first
token about 2% sooner (up to 4% at 8k) and decode under 1% faster, since decode's exchanges are small.

## Starting it

Start both servers with the same model, dump and flags (a flag that changes decoding, such as `--no-drafts`, goes on
both), rank 1 first or within five minutes of each other; each waits for the other before it loads on:

```bash
FZ_LANE=1 FZ_GDN=2 MCDMA_FABRIC_QOS=1 TF_FLASHNEXT_DUMP=$HOME/fn-dump \
  zig-out/native/bin/tensorfold-native serve ~/models/flash-next-6bit --name flash-next \
  --speed-up rank1.json --temperature 0 --no-thinking --dashboard
```

```bash
FZ_LANE=1 FZ_GDN=2 MCDMA_FABRIC_QOS=1 TF_FLASHNEXT_DUMP=$HOME/fn-dump \
  zig-out/native/bin/tensorfold-native serve ~/models/flash-next-6bit --name flash-next \
  --speed-up rank0.json --temperature 0 --no-thinking --dashboard
```

Send requests to rank 0. Rank 1 refuses requests of its own and runs rank 0's as they come; a stop string or a
cancel on rank 0 ends both Macs on the same round. Stopping rank 0 ends rank 1's part; stop rank 1's server
afterwards. If the link fails, both servers end the request with an error within about ten seconds instead of hanging.

## Limits

- Flash Next 6-bit on the replay engine only, one reply at a time, greedy.
- Two Macs. A model bigger than one Mac needs pipeline mode, which this is not.
