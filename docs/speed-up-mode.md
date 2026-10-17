# Speed-up mode: Flash Next on two Macs

Speed-up mode serves Qwen3.8 Flash Next from two Macs at once. Each Mac holds the whole model, and every prompt chunk
and every decode round is split between them over a Thunderbolt 5 cable, with [MCDMA](https://github.com/ashhart/MCDMA)
carrying the link. One Mac, rank 0, serves the OpenAI-compatible API. The other, rank 1, runs each request beside it.

Both Macs produce the same reply, and the reply is bit-identical at every draft depth. The engine drafts a few tokens a
round and keeps only the ones the model would have produced itself. A reply can still differ from a single Mac's,
because the two halves of one projection are added in a different order at the same fp32 precision.

The server loads Flash Next straight from its checkpoint, with no dump to make and no environment variable to set.
On our two M5 Ultras, set up as below, `"draft": false` replies equalled drafted ones and both Macs gave the same reply,
idle and with 36 busy threads on the first Mac.

## What you need

- Two Apple silicon Macs, each with enough memory for the whole model. Flash Next 6-bit is 158 GB of weights. On our
  two M5 Ultra Mac Studios, 256 GB each, each Mac's server takes 172 GB once loaded. Keep each server at or under 70%
  of its Mac's memory, 179 GiB on a 256 GB Mac, so macOS and everything else keep theirs. The server sizes its prompt
  cache to fit under that line, about 5 GiB on these Macs. A 192 GB Mac is too small.
- A Thunderbolt 5 cable straight from one Mac to the other.
- macOS 26.2 or later, which ships Thunderbolt RDMA. Ours run macOS 27.0.
- Xcode with its Metal toolchain and Zig 0.17.0, to build the server.

Macs with less memory, such as 64 GB Mac minis, can't run Flash Next in this mode today, since every Mac holds the whole
model. One such Mac can serve Nemotron 3.5 Lightning 30B-A3B 4-bit, 17 GiB, on the Zig engine: see
[ZIG-PREVIEW.md](../ZIG-PREVIEW.md). Spreading one model's layers over several smaller Macs needs a pipeline mode, which
isn't built.

## Setup

Do steps 1 to 4 on both Macs.

### 1. Build the server

Install Zig 0.17.0 from [ziglang.org](https://ziglang.org/download/): `zig-aarch64-macos-0.17.0.tar.xz`, sha256
`b607e9b9234790a008116ae5bdb71c6243b84b9fb42a53a9e70fde41c06c536a`. The build refuses any other version.

```bash
git clone https://github.com/ashhart/TensorFold.git
cd TensorFold
zig build native
```

That writes `zig-out/native/bin/tensorfold-native`.

### 2. Download the model

```bash
hf download TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP --local-dir ~/models/flash-next-6bit
```

`hf` comes with `pip install huggingface_hub`.

### 3. Set up MCDMA

Follow MCDMA's [README](https://github.com/ashhart/MCDMA) and its
[fabric guide](https://github.com/ashhart/MCDMA/blob/main/docs/fabric.md) for Thunderbolt links between two Macs. In
short, enable RDMA from Recovery with `rdma_ctl enable` on both Macs, build the fabric library with `make -C rpc`, which
writes `build/rpc/libmcdma-fabric.dylib`, and note each Mac's address on the cable. Thunderbolt port `enN` has the RDMA
device `rdma_enN`. We run it on two M5 Ultras over Thunderbolt 5 on macOS 27.0, with one cable and with two bonded.

### 4. Write the settings files

Each Mac gets a JSON file with its rank, the MCDMA library and the link to the other Mac. On rank 0:

```json
{"rank": 0, "library": "/path/to/MCDMA/build/rpc/libmcdma-fabric.dylib",
 "links": [{"peer": 1, "device": "rdma_en4", "via": "en4/192.0.2.2", "port": 7490, "name": "speedup"}]}
```

On rank 1, the same with `"rank": 1`, `"peer": 0` and rank 0's address on that cable:

```json
{"rank": 1, "library": "/path/to/MCDMA/build/rpc/libmcdma-fabric.dylib",
 "links": [{"peer": 0, "device": "rdma_en4", "via": "en4/192.0.2.1", "port": 7490, "name": "speedup"}]}
```

`device` is the cable's RDMA device. `via` is that port's interface and the other Mac's IPv4 address on the cable, or
its IPv6 link-local address. `port` is a UDP port both Macs keep free. The addresses above are examples. Two bonded
cables are covered in the [speed-up recipe](recipes/speed-up-mode.md).

### 5. Start rank 1, then rank 0

Both Macs run the same command with their own settings file. Start rank 1 first, then rank 0 within five minutes:

```bash
zig-out/native/bin/tensorfold-native serve ~/models/flash-next-6bit --name flash-next \
  --speed-up rank1.json --temperature 0 --no-thinking --dashboard
```

On rank 0, pass `--speed-up rank0.json` instead, and `--host 0.0.0.0` if other machines will connect, with `--api-key`
in that case.

Run both servers attached to a terminal session, or allow `tensorfold-native` under System Settings > Privacy &
Security > Local Network. macOS asks each program for Local Network access before it can reach the other Mac. A server
started detached, for example with `nohup` over SSH after the session closes, can't ask, so macOS drops its packets and
the ranks stop with `ConnectFailed` even though `fabric-check` passes.

- Earlier versions of this guide made a dump with Python tools and set `TF_FLASHNEXT_DUMP`, `FZ_LANE=1` and
  `FZ_GDN=2`. Leave them all out. The server needs no dump, and `FZ_LANE=1 FZ_GDN=2` decode 2 to 5% slower than the
  default kernels, on one M5 Ultra and on the pair, with the same replies.
- The server runs two threads at user-interactive priority, the highest a macOS program gets: MCDMA's, which places
  each message that arrives, and its own, which sends each one. For MCDMA's thread it sets `MCDMA_FABRIC_QOS=1` unless
  you set that variable yourself. At default priority a busy Mac slows every round. With 36 busy threads on the first
  Mac, `"draft": false` replies decoded at 65 tokens a second with `MCDMA_FABRIC_QOS=0`, and at 105 by default, as fast
  as on an idle pair.
- The prompt cache keeps conversation states between requests, so the next turn of a chat reads only its new tokens.
  Rank 0's cache decides and rank 1 keeps its halves under the same names. By default each Mac gives it what 70% of
  its memory leaves past the loaded server, less 2 GiB for prompt buffers, up to 16 GiB, and logs that at start, for
  example `prompt cache: 5.2 GiB from 5.2 GiB free under the 70% cap`. `--prompt-cache-gib N` sets another size up to
  what is free and `0` turns it off. A larger size is refused with the numbers, unless you add
  `--prompt-cache-over-cap`.
- `--temperature 0` matches the engine, which decodes Flash Next greedily and refuses requests with a higher
  temperature.
- `--dashboard` serves a live page at `/dashboard`.

Flags that change decoding, such as `--no-drafts` or `--no-thinking`, go on both Macs.

### 6. Connect a client

Point any OpenAI-compatible client at `http://RANK0:8888/v1` with the model name `flash-next`, where RANK0 is rank 0's
address. Send requests only to rank 0, since rank 1 refuses its own.

## Measured speeds

On our two M5 Ultra Mac Studios, 256 GB each, one Thunderbolt 5 cable, macOS 27.0, started with the commands above.
Greedy, thinking off, 256-token replies, decode in tokens a second, the mean of two runs:

| Prompt | 1k tokens | 8k tokens |
| --- | --- | --- |
| Chat | 179 | 186 |
| Code | 247 | 200 |
| Edit a file | 390 | 345 |

Edits are fastest because the reply copies long runs from the prompt. The first token came 0.25-0.30 s after a 1k prompt
and 1.19-1.29 s after an 8k one. Cold prompts ran at 7.8k to 7.9k tokens a second at 32k and 64k tokens. With
`FZ_LANE=1 FZ_GDN=2` the same build decoded 2 to 5% slower in every cell, with the same tokens.
These were measured on our machines. They aren't a promise for yours.

## Known limits

- One reply at a time. Requests queue, so several clients share one stream's speed. Running several streams in each
  round is in progress.
- A cold prompt costs the same with the prompt cache on or off: 8k to 64k-token prompts' first tokens came within
  2.2% of each other on our Macs. A follow-up turn that adds about 280 tokens to a 7-10k-token conversation got its
  first token in 0.187-0.202 s.
- The first prompt of a new size compiles some kernels. A 9.9k-token prompt took 3.6 s the first time and 1.3 s after.
- A drafted request right after a `"draft": false` one decodes slower: on our Macs, 103 against 169 tokens a second on
  a short story, and the next request is back to speed.
- When you stop rank 0, stop rank 1 too. It stops following but keeps running, and keeps a CPU core busy, until you end
  it. If a request fails partway or the link drops, restart both servers.
- Flash Next 6-bit only. We have run speed-up mode only on two M5 Ultras.
