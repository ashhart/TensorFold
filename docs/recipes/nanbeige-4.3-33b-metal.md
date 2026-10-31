# Nanbeige4.3-33B, native Metal

This recipe covers text generation with [`Nanbeige/Nanbeige4.3-33B`](https://huggingface.co/Nanbeige/Nanbeige4.3-33B)
(model_type `llama`: a dense 33B Llama-architecture model, 512K context, thinking and tool calls in its chat template)
on the Metal engine, served from the bf16 release as published: no conversion, nothing lost. It was brought up on an
M5 Ultra (256 GB); other Macs are untested. The family admits only the checkpoints checked against Hugging Face, by the
geometry their configs name (`qualified` in `zig/src/families/llama/config.zig`: Nanbeige4.3-33B); another `llama`
checkpoint is refused by name until it has been checked the same way. Prompts run on Metal 4 tensor units, so it needs
an M5 or later: M1 to M4 Macs are refused by name at load.

## Build and serve

```sh
hf download Nanbeige/Nanbeige4.3-33B --revision c1093c501cc6460427c0ac634db7223d0450c0ee \
  --local-dir models/nanbeige4.3-33b
zig build native -Doptimize=ReleaseFast -Dcpu=native
zig build grammar -Dxgrammar=PATH_TO_XGRAMMAR_v0.2.8 -Dcpu=native   # structured output
LLAMA_LOAD_LIMIT_GB=150 zig-out/native/bin/tensorfold-native serve models/nanbeige4.3-33b --name nanbeige --port 8888 \
  --context 131072 --parallel 2 --prompt-cache-gib 8
# with Nanbeige4.1-3B drafting from its bf16 release (the same replies, faster: see Drafter below)
hf download Nanbeige/Nanbeige4.1-3B --revision 9c4555c37921c0982af8fffeee1e0a6e6b953c4f --local-dir models/nanbeige4.1-3b
LLAMA_LOAD_LIMIT_GB=150 zig-out/native/bin/tensorfold-native serve models/nanbeige4.3-33b --name nanbeige --port 8888 \
  --context 131072 --parallel 2 --prompt-cache-gib 8 --drafter models/nanbeige4.1-3b --drafter-bits 0
```

`LLAMA_LOAD_LIMIT_GB` caps weights, scratch and caches together (75% of RAM when unset). The weights take 61.7 GiB; a
stream's caches take 208 KiB a position (every one of the 52 layers keeps bf16 keys and values), so a 131,072-token
stream holds 26 GiB and the full 524,288-token context 104 GiB: 150 GB holds two 131,072-token streams, or one of
262,144; the whole context needs a Mac with the weights and 104 GiB free (192 GB, the 75% default on a 256 GB Mac,
holds one 524,288-token stream). Thinking is on by default, as in the chat template; a request sends
`"chat_template_kwargs": {"enable_thinking": false}` to turn it off. The server forwards only the thinking switch to
the template: earlier turns' reasoning stays out of the prompt (the template's `preserve_thinking` default) and tool
calls take the template's XML form (its `tool_call_format` default), which the server parses.

## What the family is

`zig/src/families/llama/` follows the Spark-X2.6 Metal family, with HF's `modeling_llama` as the reference:

- The reference's rounding points, its PyTorch code in bf16: the residual stays bf16 and each add rounds to it; the
  norm is `bf16(w * bf16(x * rsqrt(mean(x^2) + eps)))` with the mean in fp32; RoPE is `bf16(bf16(x * c) + bf16(o * s))`
  with the fp32 angle's cos and sin rounded to bf16 as HF rounds them, pairs (d, d + 64); the MLP is
  `bf16(bf16(silu(g)) * u)` (`kernels/metal/llama/glue.metal`).
- Projections over the release's bf16 rows, read in place from the mapped shards (no copies, no stacking): prompt
  chunks of up to 2,048 rows on `gemm_nax`'s tensor-unit GEMM (`gemm.metal`); decode windows on `ll_matvec`, a
  simdgroup an output row, fp32 sums rounded once as torch's bf16 linear rounds them. q, k and v write side by side
  into one row, gate and up into another.
- Attention: prompt chunks on the tensor units (Gemma 4's prompt attention over a position-major cache,
  `attention_nax.metal`); decode rows on sdpa_vector lanes over key blocks at absolute positions, so a row's bits
  never depend on the rows beside it.
- The lane host's shared rounds, a lone stream's steps drawn on the GPU a step ahead (argmax, or `tf_sample_full`),
  copy drafts and the prompt cache.
- `--drafter`: a smaller Llama on the same tokenizer (`drafter.zig`) as the lane core's priced copy proposer, its
  drafts verified by the model's windows like any copy, so replies keep the model's bits.

Every kernel is registered with the kernel contract (`zig/tests/contract/metal_llama*.zig`, the roll call in
`contracts.zig`): the glue, the six row-projection forms, the four GEMM forms and the prompt attention at both served
geometries. Each kernel group has a planted control (`metal_llama_controls.zig`: one edit of its source, which its
gate must report). `tf-kernel-contract --only llama` on the M5 Ultra:

```
SUMMARY 21 kernels: 21 pass gates 1-4, 0 fail, 0 known findings; 12 controls: 12 red as planted, 0 missed; 0 limit problems; gates 468447 ms, limits 140 ms
```

## Measured (M5 Ultra, the bf16 release, 26 Oct 2026)

Against Hugging Face's `modeling_llama` on the release (transformers 4.57.1, bf16 on the CPU): 1,500 positions
teacher-forced, mean KL 0.0015, top-1 99.1%; 64 positions after a 9,000-token prompt, mean KL 0.0013, top-1 100%. Six
greedy chats of 64 tokens (code, prose, Chinese, a document, with and without thinking): three equal HF's token for
token; the other three part at a position where the engine's two best logits are the same bf16 value. One-row steps
and windows of 7 or 16 rows give the same logits bit for bit after prompts of 64 and 1,500 tokens.

`tools/zig/llama_teacher_judge.py` judges every teacher-forced position by the repository's near-tie rule
(`qwen27_margins.py`: the same two contenders on both sides, each margin within one bf16 step) and exits 1 on anything
else. Against HF's bf16 run, 13 of the 1,500 positions differ and two are red: at 425 and 1370 three tokens sit within
one bf16 step, so the two sides' contenders differ. `tools/zig/llama_f64_reference.py` runs the release in float64
(its bf16 weights cast up exactly, HF's arithmetic, one rounding at the end) and settles them:

- Against the float64 pass, the engine passes the rule: 1,483 of 1,500 positions equal, all 17 differences near-ties.
  HF's bf16 run does not: 1,488 equal, and at 425 it rounds two contenders a step high (red).
- At 1370 the exact logits are `it` 388 38.1266, `no` 754 38.1037 and `the` 269 38.0106: within 0.116 of each other,
  under half a bf16 step (0.25 at 32-64). 388 lies 0.0016 above the rounding midpoint 38.125, so rounded once it is
  38.25 and the others 38.0. HF's bf16 run gives 754 a step too high (38.25); the engine gives 388 a step too low
  (38.0).
- Layer by layer at 1370 (the engine's forward stopped after each layer count by a local probe, not in this PR; HF's
  `output_hidden_states`), the engine's residual and HF's drift from the float64 one alike (0.3% after the first layer,
  1.8% after the 51st; 0.80% and 0.82% on average over the layers). The engine's largest excess, 1.5-1.7 times HF's at
  layers 20-22, is channel 4086, a massive activation near 427 where a bf16 step is 2: both hold 430 after layer 19,
  and at layer 20 HF's add lands on 428 while the engine's stays at 430.

Every position where the engine's top token differs from HF's or from the float64 pass's (row n predicts token
n + 1; a tie goes to the lower id; logits listed for each side's top-1 tokens in the order of the first three columns):

| position | float64's top-1 | HF's | engine's | float64 (rounded once) | HF | engine | against HF | against float64 |
|---:|---|---|---|---|---|---|---|---|
| 22 | `en` 271 | `en` 271 | `see` 3574 | 20.625, 20.625 | 20.75, 20.625 | 20.625, 20.75 | near-tie | near-tie |
| 64 | `pickle` 82922 | `json` 5174 | `json` 5174 | 31.625, 31.5 | 31.625, 31.625 | 31.75, 31.75 | equal | near-tie |
| 83 | `the` 269 | `the` 269 | `Python` 9120 | 26.25, 26.25 | 26.25, 26.25 | 26.25, 26.375 | near-tie | near-tie |
| 85 | `developed` 3260 | `developed` 3260 | `-` 152369 | 35.5, 35.5 | 35.5, 35.5 | 35.5, 35.75 | near-tie | near-tie |
| 166 | `"))` 9441 | `";` 2651 | `"))` 9441 | 36.5, 36.25 | 36, 36.25 | 36.5, 36.25 | near-tie | equal |
| 171 | `\"` 12028 | `[\` 11752 | `[\` 11752 | 31.75, 31.625 | 32, 32.25 | 32.25, 32.25 | equal | near-tie |
| 224 | `('` 1091 | `(` 152374 | `('` 1091 | 39.75, 39.75 | 39.5, 39.75 | 39.5, 39.5 | near-tie | equal |
| 235 | `.` 152361 | `,` 152363 | `,` 152363 | 45.5, 45.5 | 45.25, 45.5 | 45.25, 45.5 | equal | near-tie |
| 357 | `separators` 118153 | `indent` 34489 | `indent` 34489 | 37, 36.75 | 36.75, 36.75 | 36.75, 36.75 | equal | near-tie |
| 401 | `(` 152374 | `(` 152374 | `({'` 50468 | 39.5, 39.25 | 39.5, 39.25 | 39.25, 39.25 | near-tie | near-tie |
| 425 | `1` 152367 | `{` 152405 | `1` 152367 | 39.75, 39.75 | 39.75, 40 | 40, 40 | bug | equal |
| 586 | `encoding` 13571 | `representation` 7443 | `encoding` 13571 | 29.75, 29.625 | 30, 30 | 30, 29.875 | near-tie | equal |
| 1234 | `file` 2408 | `a` 152346 | `a` 152346 | 35, 35 | 35, 35.25 | 35, 35.25 | equal | near-tie |
| 1295 | `an` 284 | `a` 261 | `a` 261 | 44, 43.75 | 43.75, 43.75 | 43.75, 43.75 | equal | near-tie |
| 1302 | `If` 3934 | `If` 3934 | `` `` `` 2283 | 32.25, 32 | 32, 31.875 | 32, 32 | near-tie | near-tie |
| 1304 | `indent` 28799 | `indent` 28799 | `allow` 13471 | 37.25, 37 | 37.25, 37 | 37, 37 | near-tie | near-tie |
| 1352 | `the` 269 | `\n` 13 | `\n` 13 | 46.25, 46 | 46, 46 | 46, 46 | equal | near-tie |
| 1370 | `it` 388 | `it` 388 | `the` 269 | 38.25, 38 | 38.25, 38 | 38, 38 | bug | near-tie |
| 1381 | `skipped` 61065 | `disabled` 18191 | `disabled` 18191 | 41.5, 41.25 | 41.25, 41.25 | 41.25, 41.25 | equal | near-tie |
| 1384 | `` `` `` 16411 | `` `` `` 16411 | `circular` 13380 | 30.875, 30.75 | 30.875, 30.75 | 30.75, 30.75 | near-tie | near-tie |
| 1447 | `instead` 62222 | `instead` 62222 | `with` 3629 | 35.5, 35.25 | 35, 34.75 | 34.5, 34.5 | near-tie | near-tie |

A device receipt with no other model server or GPU job on the Mac (28 Oct 2026, this branch's build; a CPU float64
reference run, one layer's weights at a time, overlapped its first minutes; greedy, 128-token replies,
`return_token_ids`, `--context 32768 --parallel 2`): the 1,500-position logits are bit-identical to the ones judged
above, and the six greedy chats to the earlier run; `tf-llama-run ... exact` finds 0 differing rows after the 64- and
1,500-token prompts at windows of 16 and 7 rows, and reports its planted row each time. Six prompts sent two at a time
return the same token ids as each one alone. Three chats whose first turn carries an 8,500-token log: the second turn
resumed from the prompt cache (8,521-8,528 of 8,554-8,568 prompt tokens kept) returns the same ids as on a server
without the cache. No choice differed, so there are no margins to list.

Served through the recipe harness (an idle 97 GiB server of another model was loaded beside it, so memory readings
include it), `--context 131072 --parallel 2`, greedy 256-token replies:

| streams | prose | code | structured | list | first token | a round |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16.4 | 16.3 | 16.4 | 16.4 | 0.23-0.26 s | 61.4 ms |
| 2 | 31.8 | 28.0 | 31.7 | 31.7 | 0.34-0.38 s | 66.4 ms |

Tokens a second, all streams together. One stream reads the 61.7 GiB of weights each step; two streams read them once
for both rows. Sampled replies (temperature, top-p, seeds) decode at 16.2-16.3 tokens/s, the same as greedy.

| cold prompt | 2,109 | 8,254 | 32,827 | 61,502 | 131,132 tokens |
|---|---:|---:|---:|---:|---:|
| tokens a second | 1,259 | 1,313 | 1,074 | 866 | 563 |
| time to first token | 1.68 s | 6.29 s | 30.6 s | 71.0 s | 233 s |

A fresh server, each prompt starting with words no earlier one did; the 131,132-token prompt on a
`--context 262144 --parallel 1` server, the others on the 131,072-token one. With thinking on, within 4%. A multi-turn
chat's third turn, resumed from 5,493 of its 5,649 kept tokens, starts in 0.34 s against 4.5 s cold, with the same reply.

- Structured output (json_object, json_schema, regex, choice, grammar) 18/18, drafted == plain 12/12; tool calls
  (auto, required, named, thinking, streamed) 5/5; seeded sampling repeats 9/9.
- A 60-turn soak (3 chats of 20 turns beside a decoding stream) and 12 concurrent replies each equal the request
  alone; streaming 4/4, multi-turn recall 8/8, stop strings 3/3, thinking 10/10, max_tokens 3/3.
- A passphrase at 10%, 50% and 90% of a 128,550-token prompt comes back exactly on the 131,072-token server.

## Performance options

Each keeps the replies' bits.

- **Streams (`--parallel`).** One stream decodes at 16.4 tokens/s, bound by reading the weights each step; two give
  28-32 tokens/s in all (66 ms a round against 61), each reply equal to its run alone. The row kernel holds up to 32
  rows a launch; the caches set the count (26 GiB a 131,072-token stream).
- **Context and memory (`--context`, `LLAMA_LOAD_LIMIT_GB`).** 208 KiB a token. Cold prompts run 1.1-1.3k tokens/s
  to 33k tokens, 866 at 61k and 563 at 131k (attention grows with the prompt).
- **Prompt cache (`--prompt-cache-gib`).** A resumed turn starts in 0.34 s against 4.5 s cold; kept states give way to
  streams that need the room.
- **Copy drafts.** On from the lane core, replies equal to drafts off; they accepted 0-30% of their tokens on the
  harness's prompts and gave no measurable decode gain (drafted 16 tokens/s, undrafted 16).
- **Sampling.** Draws run `tf_sample_full` over the whole 166,912-token vocabulary (the 1,024-candidate kernel is
  compiled for a 131,072-token vocabulary): sampled replies decode at the greedy speed.
- **Not offered.** No int8 or fp8 cache, no fast-math, no reduced-precision sums, no quantized weights.

## Drafter: Nanbeige4.1-3B

[`Nanbeige/Nanbeige4.1-3B`](https://huggingface.co/Nanbeige/Nanbeige4.1-3B) (revision
`9c4555c37921c0982af8fffeee1e0a6e6b953c4f`, Apache-2.0; model_type `llama`, 32 layers, hidden 2,560, 20 query and 4 key
heads of 128) shares the 33B's tokenizer: the same 166,100 ids, merges and added tokens, the same bos and eos (its
`vocab_size` pads to 166,144, the 33B's to 166,912). The family admits it as a drafter only, served from its bf16
release as published (7.33 GiB); `--drafter-bits 4` and quantized drafter checkpoints are refused by name.

- **The proposer.** `--drafter` loads the smaller Llama on the served model's queue and registers it as the lane
  host's copy proposer (`LaneHost.proposers`), priced, so a round takes its drafts as given. The model's window
  verifies them row by row and keeps the prefix its own draws agree with, as it keeps any copy draft: replies are the
  model's whatever the drafter says.
- **One budget.** The streams' caches and the drafter's (64 KiB a token) share one ledger (`budget.Ledger`). A
  stream's admission (`budget.admitStream`) checks its peak first (its caches and its drafter cache's growth, the old
  state's copy included), trims the kept prompt states to what that peak leaves, and only then allocates; a freed
  drafter cache keeps its tokens for the next request sharing its prefix, counted, and gives way first.
- **Drafting.** Before each round the drafter's cache is synced to the stream's context by their longest common
  prefix, the new tokens are fed, and the drafts are greedy steps chained on the GPU. The depth comes from measured
  rates: the stream's decayed acceptance, a chained step's time (8.5-8.9 ms) and a line fitted to the round's time by
  its window's rows. With several streams in a round, each stream drafts the depth with the most tokens a millisecond
  for the whole round (`pricing.depthShared`), nothing only when drafting costs the round more than it lands.

Checked with `tf-llama-run <33B> <tokens> draft <drafter>,<n>`, which decodes greedily through the drafter's windows:
on eight chat prompts (128 tokens, thinking on and off) and after a 1,500-token prompt (256 tokens), the drafted
tokens equal the plain greedy tokens exactly (28 Oct 2026, one model process on the Mac):

| prompt | plain | drafted | tokens a round | drafts kept |
|---|---:|---:|---:|---:|
| Fibonacci function, thinking off | 15.8 tok/s | 40.7 | 4.54 | 99 of 127 |
| Fibonacci function, thinking on | 15.8 | 22.0 | 1.81 | 57 of 142 |
| why the sky is blue, thinking off | 15.8 | 26.8 | 2.44 | 75 of 151 |
| why the sky is blue, thinking on | 15.9 | 22.8 | 1.88 | 60 of 125 |
| Hangzhou in two sentences (Chinese), thinking off | 15.9 | 21.1 | 1.73 | 54 of 137 |
| Hangzhou in two sentences (Chinese), thinking on | 15.9 | 24.1 | 2.02 | 64 of 130 |
| 9.11 or 9.8, thinking off | 15.8 | 31.1 | 3.31 | 90 of 167 |
| 9.11 or 9.8, thinking on | 15.8 | 25.9 | 2.42 | 75 of 158 |
| 1,500 tokens of Python's json docs, 256 tokens | 16.1 | 27.8 | 2.68 | 160 of 332 |

Served (`--context 32768 --parallel 2 --prompt-cache-gib 8 --drafter ... --drafter-bits 0`, fresh servers): eight new
prompts sent two at a time (greedy, 256 tokens, thinking off) decode together at 30.1-40.9 tokens/s (36.1 on average)
against 26.6-30.1 (29.0) without `--drafter`, every reply's token ids equal to the undrafted one. Streams joining,
leaving, queued, refused (HTTP 400) and cancelled each return the token ids of their plain solo run, and a reply alone
after a cancel is unchanged; in rounds shared from the first token to the last, 199 drafts were offered and 125 kept.
The drafter's cache is fed the prompt on the first round, after the model's own prompt pass, so the first token keeps
its time.

## Not done

No quality suite was run. The family has no draft head of its own: drafts come from `--drafter`'s smaller Llama or
the lane core's copies. Quantized drafters are refused by name: the drafter is served from its bf16 release.
Logprobs are refused by name: the release's tokenizer decodes through a `Strip` step, which the engine's tokenizer does
not count as token-local, so the server cannot give a token's bytes on its own.
Only the M5's tensor units were checked; M1-M4 Macs are refused at load, and smaller RAM is untested. Other `llama`
checkpoints, tied heads, other head sizes, Llama 3's scaled RoPE and biased projections are refused at load rather than
served untested.

## Licence

Nanbeige4.3-33B is published under Apache-2.0 at the revision measured here
(`c1093c501cc6460427c0ac634db7223d0450c0ee`, the licence in its model card).
