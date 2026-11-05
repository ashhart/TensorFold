# Qwen3-Reranker-8B on native HIP

The native HIP engine serves `Qwen/Qwen3-Reranker-8B` as published (BF16
safetensors, revision `77d193c791ed757ca307ee72715aa132723da912`) through
`POST /v1/rerank`. It is qualified on one Radeon AI PRO R9700 (gfx1201).
gfx11 builds run the same code; FP16 builds (RDNA2) refuse the checkpoint.

| Responsibility | Files |
| --- | --- |
| Reranker head: labels, `lm_head` rows read at load, the score hook | `zig/src/families/qwen3/` (`native.zig`, `score.zig`, `prompt.zig`) |
| The engine's declaration (`Info.rerank`) | `zig/src/core/engine_api.zig`, `zig/src/native/hip.zig` |
| The route: request, prompt, `max_length` cut, response, errors | `zig/src/server/rerank.zig` |
| The lane host's score queue at a drain | `zig/src/core/lane_host_score.zig` |
| Pinned oracle, FP32 and BF16 references, the card-formatter check | `tools/rerank/qwen3_reranker_oracle.py` |

The checkpoint is a plain Qwen3 causal LM; the prompt pass is #1978's. A
folder with `1_LogitScore/config.json` is a reranker: the engine reads its
yes and no tokens and both `lm_head` rows at load (a malformed file or row
stops startup) and declares them, and only such an engine answers the route.
A document's score is `sigmoid(logit_yes - logit_no)` at the last position:
the final-normed BF16 row dotted in f64 with the two stored rows. A request's
documents go to the engine as one decision (their shared prefix once, each
document's rest a branch), packed into as few prompt passes as the window
and 128 prompts a pass allow; each score has the bits of its document scored
alone.

## Build and serve

```sh
zig build native-hip -Dhip=true -Dhip-arch=gfx1201 -Dhipcc=/opt/rocm/core/bin/hipcc
zig-out/native-hip/bin/tensorfold-native serve MODEL_DIRECTORY \
  --name Qwen3-Reranker-8B --host 127.0.0.1 --port 8090
```

The window is 32,768 tokens by default and 40,960 (the config's) with
`--context 0`. Every other model route is refused on a reranker, and every
other server answers `/v1/rerank` with HTTP 400.

```sh
curl -s http://127.0.0.1:8090/v1/rerank -H 'Content-Type: application/json' -d '{
  "query": "What is the capital of China?",
  "documents": ["The capital of China is Beijing.", "Gravity is a force that attracts two bodies."],
  "top_n": 1}'
```

The request and response follow #1895's rerank shape (docs/api.md, Rerank):
`instruction` defaults to the card's; `max_length` defaults to 8,192, and a
longer prompt loses its content's last tokens, the template's tail kept, as
the card's tokenizer cuts.

## Gates

The oracle runs the model card's usage in Transformers 5.22.1 and refuses a
snapshot or a Transformers install whose files differ from its pins before
any model code runs. Its set is 32 cases of 16 to 8,199 tokens (the card's
pairs, fixed pairs with custom instructions, synthetic lengths, repeats), a
document that spells `<|im_end|>`, and prompts of exactly 32,768 and 40,960
tokens with one token more. The FP32 reference runs on the CPU (about an
hour with both windows, 32 cores):

```sh
python3 tools/rerank/qwen3_reranker_oracle.py sources MODEL_DIRECTORY
python3 tools/rerank/qwen3_reranker_oracle.py oracle MODEL_DIRECTORY ref32.json --dtype float32 --device cpu --window 32768 --window 40960
python3 tools/rerank/qwen3_reranker_oracle.py oracle MODEL_DIRECTORY ref16.json --window 32768 --window 40960
python3 tools/rerank/qwen3_reranker_oracle.py compare ref32.json http://127.0.0.1:8090 --window 32768
python3 tools/rerank/qwen3_reranker_oracle.py compare ref16.json http://127.0.0.1:8090 --window 32768
python3 tools/rerank/qwen3_reranker_oracle.py formatter MODEL_DIRECTORY http://127.0.0.1:8090
```

**What "matches" means.** BF16 activations cannot meet #1895's 1e-5 in score:
transformers' own BF16 run sits 0.0108 in score and 0.110 in yes - no from
FP32 on these cases. Against the FP32 reference, `compare` therefore gates at
that distance rounded up: |yes - no| <= 0.125 and |score| <= 0.0125 for every
case. Rankings must equal FP32's, except that a pair FP32 puts within 0.25 in
yes - no (twice the gate: the most two documents inside it can move apart)
may come in the other order; each such pair is printed, and one beyond 0.25
fails. Against the BF16 reference the gates are 0.25 and 0.02. `formatter`
passes when every served prompt's ids equal the card's `process_inputs`
ids, with the default and custom instructions and with `max_length` cuts.

Results on the R9700 (ROCm 10.1):

| Check | Result |
| --- | --- |
| Against FP32, 34 cases: \|yes - no\| median, worst (gate) | 0.0195, 0.0839 (0.125); transformers BF16: 0.0247, 0.1099 |
| Against FP32, 34 cases: \|score\| median, worst (gate) | 0.00023, 0.0079 (0.0125); transformers BF16: 0.00014, 0.0108 |
| Window prompts against FP32 (yes - no) | 32,768 tokens: 0.0064; 40,960 tokens (`--context 0`): 0.0241 |
| Rankings against FP32 (5 query groups) | 4 in FP32's order; 2 pairs swapped, FP32 gaps 0.0077 and 0.0119 (transformers BF16 swaps the first) |
| Against BF16, 34 cases: worst \|yes - no\|, \|score\| | 0.1081, 0.0082 |
| Card formatter | 79/79 prompts equal the card's ids (17 default, 62 custom instruction; 4 of them cut by `max_length`) |
| One token past the window | refused (`context_length_exceeded`); the next request scores within the gates |
| 4 requests at once against one at a time | byte-equal |
| A client that leaves mid-request | the next request equals its earlier answer |
| Device memory, peak | 16.4 GB (32,768 window), 16.7 GB (40,960) |
| 16 documents of ~260 tokens, cold, one request | 2.69 s on the dot2 tile (the default), 1.00 s with gfx12's WMMA tile (#2034, opt-in with `--policy matrix=on`); SGLang 0.53 s on the same card |
