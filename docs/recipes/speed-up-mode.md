# Speed-up mode: one model on two Macs

Speed-up mode runs Qwen3.8 Flash Next 6-bit on two Macs joined by Thunderbolt. Each Mac holds the whole model
and does half the work of every request: prompts split their rows between the Macs, and each decode round splits
its DeltaNet heads, routed experts and vocabulary head. One Mac serves the HTTP API and the other runs every
request beside it. It needs the native Zig server (`tensorfold-native`) with the Flash Next replay engine, and
MCDMA for the Thunderbolt link.

## What it gives

Measured on two M5 Ultra Macs (256 GB each) over one Thunderbolt 5 cable, greedy, 256-token replies, against
the same server build on one of the Macs:

| Prompt | Time to first token, two Macs / one | Decode, two Macs / one (tok/s) |
| --- | --- | --- |
| 1k code | 0.27 s / 0.34 s | 207 / 159 |
| 1k edit | 0.31 s / 0.38 s | 348 / 286 |
| 1k chat | 0.25 s / 0.34 s | 170 / 144 |
| 8k code | 1.30 s / 2.05 s | 168 / 154 |
| 8k edit | 1.21 s / 2.02 s | 328 / 280 |
| 8k chat | 1.12 s / 1.91 s | 181 / 140 |
| 32k code | 4.06 s / 6.95 s | 189 / 163 |
| 32k chat | 4.00 s / 6.86 s | 168 / 137 |

Prompts run 1.6 to 1.7 times as fast from 8k tokens (about 8,000 tokens a second at 32k). Decode gains less, 1.1
to 1.3 times, because each layer exchanges two partial results between the Macs (about 15 microseconds each over
one cable) and the draft head runs on both.

Both Macs produce the same reply token for token. A reply can differ from one Mac's in rare tokens: the two halves
of a projection are added in a different order, at the same fp32 precision. Prompt splitting gives one Mac's bits
exactly.

## What you need

- Two Apple silicon Macs with enough memory for the whole model on each (the 6-bit checkpoint and the replay
  engine's dump, about 180 GB at peak on each), and the same checkpoint and dump on both.
- A Thunderbolt 5 cable between them with RDMA enabled, and MCDMA's fabric library (`libmcdma-fabric.dylib`) built
  on both.
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

## Starting it

Start both servers, rank 1 first or within five minutes of each other; each waits for the other before it loads
on:

```bash
TF_FLASHNEXT_DUMP=/path/to/dump tensorfold-native serve /path/to/Qwen3.8-Flash-Next-6bit --speed-up rank1.json --no-thinking
```

```bash
TF_FLASHNEXT_DUMP=/path/to/dump tensorfold-native serve /path/to/Qwen3.8-Flash-Next-6bit --speed-up rank0.json --no-thinking
```

Send requests to rank 0. Rank 1 refuses requests of its own and runs rank 0's as they come; a stop string or a
cancel on rank 0 ends both Macs on the same round. Stopping rank 0 ends rank 1's part; stop rank 1's server
afterwards. If the link fails, both servers end the request with an error within about ten seconds instead of hanging.

## Limits

- Flash Next 6-bit on the replay engine only, one reply at a time, greedy.
- Two Macs. A model bigger than one Mac needs pipeline mode, which this is not.
