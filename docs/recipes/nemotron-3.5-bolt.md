# Nemotron 3.5 Bolt

[NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) is a
74B-parameter Mamba-2, attention and MoE model (9B active) packed to 24 GB: IQ2_XXS experts, NVFP4 Mamba projections,
Q8_0 attention and latent projections. Its `config.json` is a `nemotron_h_omni` wrapper around a `nemotron_h_puzzle`
text model (hidden 4096, 88 blocks, 240 to 496 routed experts a layer, top-9), so the native CUDA host serves it as its
own family, `nemotron_h_puzzle`, on one GB10 (sm_121). The family is `zig/src/families/nemotron/bolt_native.zig`, a lane
backend over `libbolt.so`; the HTTP server, chat template, round loop, keyed sampler and prompt cache are the engine's.

## Build and run

`libbolt.so` is built from its CUDA sources, which this repository does not carry (they are
[MiaAI-Lab's](https://github.com/MiaAI-Lab/Nemotron-3.5-Bolt-TensorFold-DGX-Spark/tree/main/engine)):

```bash
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dbolt-src=PATH/engine -Doptimize=fast install native bolt
tensorfold-native serve NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB --parallel 8 --context 262144
```

`native/lib/libbolt.so` is found beside the server (`TENSORFOLD_BOLT_LIB` names another); the backend checks its C ABI
version at load. Images need `--vision`; videos also `libtfvideo` (`zig build video`). The drafter,
[LiLiCorr](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr), is named by `BOLT_DRAFT`
(a folder) or `--no-drafts` serves one row a round. `BOLT_KV=fp8` stores the attention cache as e4m3 with a scale per
token and head (4,224 bytes a token against 8,192), and `BOLT_KV_PIN=1` maps every stream's whole window at load.

## What it serves

- Up to 16 streams (`--parallel`) on one copy of the weights; a shared round carries every stream's verify window in one
  forward. The window is the model's 262,144 tokens a stream; attention rows are mapped as a stream grows.
- LiLiCorr drafts verified in the lane rounds, seven tokens a window, drawn with the stream's keyed sampler.
- Prompt cache over the Mamba state, attention rows and drafter context (`snap_*`, any row, no lookahead).
- Images through the RADIO tower (`--vision`): each prompt piece encodes the images it crosses into the rows buffer, so a
  request holds as many images as fit the window (`--vision-max-images`, up to 64). Videos through the same tower: frames
  sampled at 2 a second (1 above 720 rows, at most 128), two to a tubelet through `video_embedder`, each tubelet labelled
  "Frame i sampled at t seconds and frame j ..." as the checkpoint's processor writes it (`--vision-max-videos`).
- Structured output and forced tool calls: `response_format` (`json_schema`, `json_object`), `guided_*` fields and
  `tool_choice` `required` or a named function. The server compiles the grammar (`libtfgrammar`, `zig build grammar`) to
  per-row allowed-token bits; the backend redraws the rows of a verify window among those tokens (a masked top-k on the
  GPU, `bolt_mask_rows`), hands the grammar its held drafts, and draws the first token again under the first mask.
- Not served: logprobs, speculative trees, int8 or int4 caches, `--slide`.

## Exactness

A stream's rounding is the same alone, among other streams and in a verify window of any width. Three things made it so:

1. **Attention chunks follow the stream's own length.** A verify window merges partial softmaxes over chunks of keys. The
   chunk is `win_chunk(base)`: 128 keys, doubled until 32 chunks reach the end of the sequence and its window. Before, the
   chunks followed the number of streams in the block (about 32 splits for one stream, 8 for four) and the window's width,
   so a stream among others rounded differently from the same stream alone.
2. **Decode GEMMs sum K in one order.** Dense GEMMs for blocks of up to 32 rows split K across four warps (interleaved
   partial sums, then added); blocks over 32 rows used a kernel that walks K in one accumulator. A stream's rounding changed
   when other streams grew the block past 32 rows. Decode and verify blocks now always use the K-split kernel (64 tokens
   a CTA); the prompt pass keeps the faster large-block kernel.
3. **The plain step is a window.** The one-token step (no drafts) runs as a one-row verify window and is kept at once, so
   it shares the arithmetic of drafted decoding.

Checks in the engine's harness (not in this repository), run with 32-token GEMM tiles: the perplexity of four passages
scored alone and in a block of 2, 3, 4, 8 and 16 streams (windows of 1, 5 and 8 rows, up to 128 rows a block) is
identical to five decimals; sixteen sequences decoded in one block equal their solo runs for 768 of 768 tokens, plain
steps and 4-row windows alike. The final 64-token tile changes only the tokens a CTA holds, not the order K is summed
in, and is covered by the `bench_concurrent.py` table below. Perplexity on 1,499 tokens is 5.286. A stream's logits are
bit-identical whatever its neighbours hold (leak test).

The prompt pass uses different kernels from decode by design; `--alone` compares decode.

## Receipts

TensorFold `46a08c18` plus this change; DGX Spark GB10, Ubuntu 24.04.4, kernel 6.17.0-1026-nvidia, driver 580.159.03,
CUDA 13.3.73; checkpoint `a326822af111290f2af5ac59ef4b5040347dabc4`, drafter `4c6a5613b6d887194a3a5493e3a18a5b3930bb86`;
`--parallel 8 --context 262144 --prompt-cache-gib 8 --vision`, bf16 cache with every stream's window mapped at load
(`BOLT_KV_PIN=1`), greedy, 256 tokens, 3 repetitions.

`tools/bench_concurrent.py --levels 1,4,8 --alone --serial`:

| Prompt | Streams | Aggregate | Each stream | Max TTFT | Concurrent equals solo (`--alone`) | Drafted equals `"draft": false` (`--serial`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code | 1 | 115.1 tok/s | 115.1 | 0.07 s | 3 of 3 | 1 of 1 |
| code | 4 | 207.9 tok/s | 54.8 | 0.45 s | 12 of 12 | 1 of 1 |
| code | 8 | 252.6 tok/s | 35.2 | 1.15 s | 24 of 24 | 1 of 1 |
| chat | 1 | 60.7 tok/s | 60.7 | 0.05 s | 3 of 3 | 1 of 1 |
| chat | 4 | 139.1 tok/s | 35.6 | 0.42 s | 12 of 12 | 1 of 1 |
| chat | 8 | 174.7 tok/s | 23.6 | 1.05 s | 24 of 24 | 1 of 1 |

Cold prefill with `tools/bench_cache.py --require-cold` (0 cached tokens, `qualified: true`), measured on the build
before the exactness changes, which touch decode blocks only (not re-run since): TTFT 1.44 s at 2,048 tokens, 5.60 s at
8,192, 24.66 s at 32,768, 55.60 s at 65,536.

Cost of the K-split kernel in the widest rounds (a shared round of N streams x R rows, harness, ms; before the change /
after): 8 x 8 rows 135.5 / 130.4, 16 x 4 rows 133.2 / 141.7, 16 x 8 rows (128 rows) 176.9 / 220.3. Rounds of up to 32 rows
are unchanged.

## Limits of this change

`libbolt` is a CUDA runtime library, not fatbins under the kernel contract, and its sources are outside this tree
(4 files, about 4,800 lines, past the lean checker's file and comment rules); they are in the
[recipe repository](https://github.com/MiaAI-Lab/Nemotron-3.5-Bolt-TensorFold-DGX-Spark/tree/main/engine), which is
private for now. It targets sm_121 only and was run on one GB10. Whether the kernels belong in `zig/kernels/cuda` under
the contract, stay a separate repository, or go through another route is the maintainers' call; the Zig side here is the
part that does not depend on that answer.
