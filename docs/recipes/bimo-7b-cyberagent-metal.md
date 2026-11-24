# Bimo-7B-CyberAgent on Metal: the bf16 release through the Llama family

[Bimo-7B-CyberAgent](https://huggingface.co/iBimo-o/Bimo-7B-CyberAgent) (iBimo-o, Apache 2.0) is a Qwen2
checkpoint: Qwen2.5-Coder-7B-Instruct fine-tuned for defensive security work, 7.6B parameters, 28 layers, 28 query
and 4 key heads of 128, a 152,064-token vocabulary, q, k and v biases, and static YaRN (factor 4 over its 8,192
trained window) to a 32,768-token context. The engine serves its bf16 release as published (revision 9c1954d8,
15.2 GB, nothing converted) through `zig/src/families/llama/`, which admits `model_type: qwen2` beside `llama`
for the checkpoints checked against Hugging Face (`config.zig`'s `qualified`). Prompts run on Metal 4 tensor units,
so it needs an M5 or later; M1 to M4 Macs are refused by name at load.

```sh
hf download iBimo-o/Bimo-7B-CyberAgent --revision 9c1954d83d1768720912a8ebdff5db1fdcfb4f87 --local-dir models/bimo-7b
zig build native
LLAMA_LOAD_LIMIT_GB=40 zig-out/native/bin/tensorfold-native serve models/bimo-7b --name bimo --port 8896 \
  --context 32768 --parallel 2 --prompt-cache-gib 4 --temperature 0.7 --top-p 0.8 --top-k 20
```

`libtfgrammar` beside the binary (`zig build grammar -Dxgrammar=<xgrammar v0.2.8 checkout>`) serves structured output
and `tool_choice: required`. The release's `generation_config.json` names temperature 0.7, top_p 0.8 and top_k 20,
which the flags above give requests that set none; its `repetition_penalty` 1.1 is not applied (the family has no
penalties). The template has no think block, so every reply is `content`; `--drafter` is refused by name, since no
drafter has been checked with this checkpoint. The lane core's copy drafts landed 1.05 tokens a round on the
harness's ruler and cost 6% (49 against 52 tokens/s on its drafts check, replies equal 12/12, engine 7662e079), so the family serves
this checkpoint without them (`copy_drafts` in its qualified entry; `--no-drafts` changes nothing).

## What the Qwen2 path adds to the family

The reference is HF's `modeling_qwen2` in bf16, whose rounding points are `modeling_llama`'s with two additions:

- **Biases.** q, k and v are `F.linear(x, W, b)`: the bias joins the fp32 sum and the result rounds once to bf16.
  Biased row kernels (`ll_matvec_bias_*`, a window's rows) take the bias as a fifth buffer and add it to the
  simd-summed total before the rounding; the unbiased `ll_matvec_*` the other checkpoints serve stay
  `core/bf16_matvec.metal`'s shared kernels, unchanged. A prompt chunk's GEMM has no epilogue, so a biased
  projection runs the same `gemm_nax` tiles storing their fp32 sums (`ll_gemm_f32_*`, `TF_GEMM_NAX_KERNEL_TO` in
  `gemm.metal`) and `ll_bias_round` adds the bias and rounds once; o and the MLP keep the bf16 forms. The scratch holds one fp32 chunk of the widest biased projection (q, 3,584 wide).
- **YaRN.** The rope table is transformers' `_compute_yarn_parameters` for 128-wide heads
  (`core/rope_yarn.zig`'s `yarnOf`, the 64-wide table unchanged word for word), and the cosines and sines carry
  the attention factor 0.1 ln(4) + 1 before their bf16 rounding. They are not taken on the GPU: at load the host
  makes the window's bf16 cosines and sines at HF's rounding points (`llama/rope.zig`: the fp32 angle, its
  correctly rounded fp32 cos and sin, times the factor in fp32, then bf16; 8 MiB for 32,768 positions, counted in
  the load), and `ll_prep_table` reads them. Of the table's 4,194,304 words, all but 19 equal transformers 5.22.2's
  `Qwen2RotaryEmbedding` (torch 2.15.0, CPU). At those 19 the scaled value sits on a bf16 tie (within 2.1e-5 of a
  step of the midpoint), torch's fp32 cos or sin is one fp32 ulp off the correctly rounded one, and the two
  words are one bf16 step apart. A test pins the 19 words and, with them patched in, the whole table's hash against
  HF's. Every other checkpoint keeps `ll_prep` as main has it. The frequency table equals transformers 5.22.2's fp32 words, pinned in a test (4.50 ignores the trained window there); the config's `rope_parameters` (transformers
  5's spelling, `rope_theta` inside it) and `rope_scaling` are both read, and a YaRN config naming any other key
  (`attention_factor`, `beta_fast`, ...) or a context that is not factor × trained window is refused.

Nanbeige4.3-33B's and Nanbeige4.1-3B's logits are bit-identical to main's on this branch (`tf-llama-run logits`
over a 2,000-token prompt, 667,648,000 and 664,576,000 bytes equal).

## Checked against the release

`tools/zig/llama_f64_reference.py` now takes a Qwen2 config (its biases and YaRN at HF's rounding points, in float64,
one rounding at the end); `tools/zig/llama_teacher_judge.py` judges each position by the repository's near-tie rule.
Three chats through the release's template (a coding task, a security question, a tool-use task with the tools
described in the system message), HF's own 64 greedy tokens each (transformers 5.22.2, bf16 on the CPU) forced
through both, the engine at 19382dafb:

| 192 positions | equal | differ | verdict |
|---|---:|---:|---|
| engine against float64 | 190 | 2 (near-ties) | pass |
| HF's bf16 run against float64 | 189 | 3 (near-ties) | pass |
| engine against HF's bf16 run | 191 | 1 (near-tie) | pass |

Greedy, the engine's 64 tokens equal HF's on the security and tool-use chats token for token; the coding chat
parts at step 36, where the engine's two best logits are the same bf16 value (20.0) and HF's differ by one step.
One-row steps and 16-row windows give the same logits bit for bit over those 64 tokens (`tf-llama-run exact`).

Served (`tensorfold-native serve`, two streams, 32,768-token window): sampled replies at the release's settings;
structured output (`json_schema`) valid; two concurrent greedy replies equal to their solo runs; a 13,090-token
second turn resumed from 13,062 kept tokens with the same reply as a freshly started server (the whole request,
up to 64 tokens, 0.99 s against 3.11 s); a 27,633-token prompt served in 5.7 s. Tool calls: the published template renders no `tools` field, so a request
whose system message describes the tools (the Hermes form the model was trained on) gets `<tool_call>` JSON that
the server parses into `tool_calls`, with `tool_choice: required` honoured; `tools` alone, without that
description, gets prose. A passphrase in filler comes back at 4k, 8k, 16k and 20k tokens and at 90% of a
27.6k-token prompt; at 10% and 50% of 27.6k the engine and HF's own run both answer with another number, the
model's behaviour, not the engine's.

## Kernel contract

The new kernels (`ll_prep_table`, `ll_matvec_bias_*`, `ll_gemm_f32_*`, `ll_bias_round`) and the prompt attention at
this geometry are registered in `zig/tests/contract/metal_llama*.zig`, the biased row kernels at Bimo's q and k/v
shapes and the unbiased ones at Nanbeige's as before, with six more planted controls (the bias dropped, the fp32 GEMM
untransposed, the rounding without its bias, the sum rounded before the bias joins it, the table read at the window's
first position, its cosines and sines swapped). `ll_prep` is main's kernel and keeps main's bound and shapes.
`ll_prep_table` reads the same bf16 table the reference rotates with, so its bound is zero: its outputs must equal the
float64 reference's bf16 words exactly.
`tf-kernel-contract --only llama` on the M5 Ultra:

```
SUMMARY 66 kernels: 66 pass gates 1-4, 0 fail, 0 known findings; 20 controls: 20 red as planted, 0 missed; 0 limit problems; gates 1012775 ms, limits 539 ms
```

## Measured

M5 Ultra (256 GB), the recipe harness's ruler, greedy 256-token replies, engine 19382dafb, `--no-drafts`, no other
model loaded (the harness's `speed` and `prefill-32k` plans):

| streams | prose | code | structured | list | first token |
|---:|---:|---:|---:|---:|---:|
| 1 | 55.9 | 55.3 | 55.6 | 55.0 | 0.07-0.08 s |
| 2 | 111.3 | 103.1 | 111.1 | 114.1 | 0.11 s |

Tokens a second, all streams together: one stream reads the 14.7 GiB of weights each step (18.1 ms a round), two
streams read them once for both rows (18.3 ms). Cold prompts: 2,270 tokens at 5,821 tokens/s, 8,834 at 6,080 and
17,587 at 5,619 (the plan's 30,720-token point records no result at this window, as on 7662e079); a 27,633-token
prompt is served in 5.7 s (above). A chat's third turn resumed from 5,493 of its 5,583 kept tokens starts in 0.10 s against
0.98 s cold, with the same reply. The harness's `quick` plan on the same build: structured 18/18 (drafted == plain
12/12), soak 60/60, concurrent 12/12, streaming 4/4, multi-turn 8/8, stop 3/3, copy 2/2, sampling 9/9; its tool
cases 4/5 (the `auto` case with two tools answers in prose with the published template, and writes a fabricated
`<tool_response>` instead of a call with the base model's tool-rendering template); its thinking cases do not
apply, and its max-tokens check is 2/3.

The recipe harness's quality suite, thinking off, on engine 7662e079 (before the rotation table; not rerun on this
build), greedy, its own prompts and graders (the recipe's
`evidence/bf16-quality-off` and `evidence/bf16-humaneval-long-32k-off`): GSM8K 180/250, MATH-500 207/500, GPQA
diamond 47/198, MMLU 182/342 (57 subjects x 6), MGSM 96/240, HumanEval 79/164 pass@1, IFEval 61/150 loose and
57/150 strict, JSON 25/30, tool calls 9/60 (the published template renders no `tools`), repetition 0/8 (repeated
4-grams 36.9% mean at greedy), needles 3/3 at 7k tokens, RULER-style long tasks 0/10 at 28k. The HumanEval
failures are mostly a doubled closing parenthesis and the repetition replies loop; both are the model's: HF's bf16
run teacher-forced over those replies gives the engine's token at every position of two 214- and 191-token
HumanEval replies and of the 1,200-token worst repetition reply, and differs once at a near-tie in a third.

## Licence

Bimo-7B-CyberAgent is published under Apache 2.0 on Hugging Face; TensorFold redistributes no weights, the server
reads the publisher's files.
