# GLM-5.3 (full model, several Macs)

The `glm53` family serves the full GLM-5.3 (`GlmMoeDsaForCausalLM`: 78 layers, 256 routed experts with top-8, DSA
sparse attention with `index_topk` 2048, and the layer-78 MTP head) across Macs that together hold its weights. It is
the full model next to [GLM-5.3-Flash](glm-5.3-flash.md). One binary, `tf-glm53`, runs one rank. The ranks split every
layer's heads, expert rows and vocabulary, and exchange fp32 partials over [MCDMA](https://github.com/ashhart/MCDMA)
inside the command buffer.

Measured on four M3 Ultra Mac Studios (256, 512, 256 and 256 GB) over Thunderbolt 5. Not measured on M5.

## Checkpoint

- The trunk: `orcarouter/GLM-5.3-MLX`, subfolder `6-bit/` (about 670 GB). Routed experts are 6-bit (down 8-bit),
  shared experts and attention 8-bit, groups of 64, F16 scales and biases. The DSA indexer, router, norms, embedding
  and head stay BF16. The quantized matmuls promote to fp32, so the engine runs fp32 activations and the reference is
  mlx-lm's fp32 path.
- The MLA absorb pack: `tools/mla_pack.py CHECKPOINT_DIR mla_pack.safetensors` saves mlx-lm's `embed_q` and
  `unembed_out` (each layer's `kv_b_proj`, re-quantized 8-bit g64), so the engine reads mlx-lm's values.
- The MTP head (optional, drafts only): the trunk drops layer 78. `tools/convert_mtp78.py` repacks it from
  `guruswami-ai/glm-5.3-mlx-mixed-4_8bit-mtp-recipe` into the trunk's layout, losslessly.
  `tools/convert_mtp78_hq.py` keeps that file's non-expert tensors and re-quantizes the routed experts from
  `zai-org/GLM-5.3-BF16` at the trunk's widths. The measurements below use this one. The verifier decides every token,
  so the head changes speed, never replies.

Keep the indexer BF16. On GLM-5.2, every 4-8-bit quant of the indexer we tried lost long context at about
3,000-4,700 tokens; BF16 held.

## Build and run

```bash
zig build tf-glm53 -Doptimize=ReleaseFast -Dcpu=apple_m1
./zig-out/bin/tf-glm53 rank0.json
```

One settings file per rank. Start every rank; each waits for the others' start flags and refuses a peer whose exchange
settings differ.

```json
{"model": "/path/GLM-5.3-MLX/6-bit", "pack": "/path/mla_pack.safetensors", "mtp": "/path/mtp78.safetensors",
 "rank": 0, "ranks": 4, "library": "/path/MCDMA/build/rpc/libmcdma-fabric.dylib",
 "links": [{"peer": 1, "device": "rdma_en4", "via": "en4", "port": 7901, "name": "r0-r1"}, ...],
 "layers": [0, 78], "cap": 3100000, "rows": 256, "kv16": true, "threads": 16, "serve": 8041}
```

- `rows`: the largest prompt block. Decode is one row; a verify block is up to 8.
- `kv16`: fp16 latent and indexer-key caches, half the memory a position. Replies were identical to fp32 caches on our
  checks. Snapshots are tagged by KV width.
- `cap`: positions every cache is allocated for, up front.
- One Mac: `"ranks": 1, "slices": N` computes N canonical slices and sums them in the same order, so its bits equal N
  Macs'. `"oracle"` compares hidden states, logits and top-k sets with an mlx-lm dump; `"prompt"` + `"max"` generates.
- `tf-glm53-ixbench` checks and times the indexer and sparse attention at long context on synthetic data.
  `tools/synth_ckpt.py` writes a three-layer synthetic checkpoint at real shapes for tests without the weights.

### Serving

`"serve": port` takes jobs over TCP, a JSON line each, and every rank must get every job:
`{"p0": first position, "tokens": [...], "max": n, "stop": [...]}`. The caches hold every position before `p0`,
so a conversation sends only its new tokens. Rank 0 answers `{"t": id}` per token, then a `done` line with timings.
`{"cancel": true}` stops a reply on rank 0, which announces `{"stopping_at": k}`; relay `{"stop_at": k}` to the others so
every rank stops after the same token. `{"save": path, "n": n}` and `{"load": path, "n": n}` write and read every
layer's first n KV rows, which hold the exact bits, so a saved conversation continues as if it never stopped. An
OpenAI-compatible front over these lines is not part of this family.

## Switches

Every switch is exact: replies stay character-identical with it on or off. Set the same values on every rank; the
start flag carries the ones that change the exchanges. The configuration we serve sets all of them.

| Switch | What it does |
| --- | --- |
| `G53_CONC=1` | Concurrent encoder, with barriers only at data dependencies |
| `G53_FWAIT=1` | The exchange's wait runs inside the sum launch: one launch fewer an exchange |
| `G53_GROWS=1` | bf16 matvecs over several rows (verify head, router, indexer, MTP projection) share each weight read |
| `G53_JOINB=1` | The attention join for blocks of up to 16 rows issues its loads 8 blocks at a time |
| `G53_ATTN4H=1` | Sparse attention four heads a threadgroup |
| `G53_IDX=3` | Indexer scores two rows a threadgroup, each key read once for both |
| `G53_KSPLIT=1` | Decode and verify rows past `G53_KSPLIT_MIN` keys (default 65,536): each rank scores a quarter of the keys and keeps its top 2,048 candidates; the ranks swap and merge them into the same picks |
| `G53_OWN_FRONT=1` | Prompt blocks of 64 rows or more: each rank runs the indexer for its quarter of the rows and the picks are all-gathered |
| `G53_COPY=3` | Copy drafts: a suffix match of at least 3 tokens proposes what followed it earlier, verified like any draft (`G53_COPY_ROW`, default 0.30, is a verify row's relative cost) |
| `G53_TOUCH=1`, `G53_KV_RESIDENT=1` | After a snapshot load, touch its pages; keep the KV and activation buffers in the residency set |
| `G53_XCHUNK=2097152` | Large partials go in fenced 2 MiB pieces, with at most `G53_XLIMIT` (4 MiB) unfenced on a link. Without it, Thunderbolt dropped an overflowing partial without an error and every rank spun |
| `G53_STALL_S=5` | A wait longer than this prints which rank and peer it waits for |

MTP drafts need the head file. The controller picks 0 to `mtp_max` drafts a round (default 3) from the acceptance it
has seen and `mtp_cost`, and probes one deeper every 16 rounds. `mtp_depth` fixes the count.

### Context past the replicated cache

Each rank holds every layer's latent cache by default: 576 values a position a layer, plus the indexer keys on the full
layers. At 2M positions that's about 190 GB a rank, more than fits beside the weights.

`G53_LSPLIT=1` splits it by layer. Layer i's latent rows, and its attention for all 64 heads, live on one rank. That
rank holds every head's `q_b`, `embed_q` and `unembed` for its layers, and sends each peer only its heads' values. The
indexer keys stay on every rank.
- `G53_LSPLIT_OWNER=r` puts every layer on rank r, the Mac with the most spare memory. With a 512 GB owner and three
  256 GB Macs, the cap goes from about 2.0M to 3.1M positions.
- `G53_LSPLIT_FROM=X` keeps positions below X on every rank. A block whose keys all lie below X runs the plain engine
  at full speed. Only blocks past X take the split, and their replies are the same.
- `G53_SCORE_ROWS` and `G53_ATTN_ROWS` (default 64 under the split) run long prompt blocks' indexer scores and attention
  in row chunks, so the score buffer and attention partials hold that many rows.

We serve `G53_LSPLIT=1 G53_LSPLIT_OWNER=1 G53_LSPLIT_FROM=300000` with `"cap": 3100000`.

## Measured

Four M3 Ultra Mac Studios, the configuration above, MTP head on. Decode is tokens a second as seen by the client.

| | |
| --- | --- |
| Decode, a short answer / a 400-token story / 400 tokens of code | 51.5 / 41.0 / 45.7 tok/s |
| Decode in a real conversation at 138K-183K positions | 36.2-42.9 tok/s |
| An 8K prompt read, needle found | 153 tok/s |
| Prompt read at 2K / 8K / 32K / 64K (fresh cache) | 180 / 159 / 150 / 143 tok/s |
| Crossing `G53_LSPLIT_FROM` (a 7.5K read past it) | 120 tok/s read, 33.8 tok/s decode, the same reply as the plain engine |
| Deepest recall | a needle at about 1,015,896 positions in a 1.03M-position conversation, 38 of 38 checks |
| Endurance | 3 h 41 min of continuous use up to 1.03M positions, no stalls |
| Decode near 1M positions | about 27-30 tok/s |

The first three rows were measured on this branch's binary, A/B against the build we serve: the same speeds and the
same reply hashes on all six long-conversation turns. The rest were measured on that served build the night before.

## Exactness checks

- One Mac with N slices against N Macs: the same hidden-state bits.
- `oracle` mode against mlx-lm's fp32 path (layers 0-9, 2,300 tokens): argmax the same at every checked position,
  hidden-state rel-L2 1.6e-7 to 4.2e-7.
- Every switch, on against off: replies character-identical on fixed prompts (short, a 250-word story, code, a 19K
  prompt), the same 8K needle, a tool call, and the same reply hashes on real long conversations.
- A fixed 70-question set (50 MMLU-Pro style, 20 short forced answers) against mlx-lm's fp32-activation reference
  split four ways: the same score.

## Limits

- Four Macs and about 670 GB of weights. The per-rank shares assume equal head and expert-row splits.
- Past `G53_LSPLIT_FROM`, decode slows to the owner's pace (about 27-30 tok/s near 1M positions on our Macs).
- Thunderbolt exchange stalls on long prompts are detected and reported (`G53_STALL_S`). The root fix, flow control or
  reconnect in MCDMA's Thunderbolt link, belongs in MCDMA.
