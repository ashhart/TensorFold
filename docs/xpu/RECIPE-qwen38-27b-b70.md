# Recipe (draft): Qwen3.8-27B on one Arc Pro B70

> **Status: draft.** This describes how our serving setup is made so that someone else can reproduce it. Items marked
> `TODO` are not recorded yet or have not been re-run from this text. Nothing here is a benchmark: numbers live in
> [README.md](README.md) (section "Numbers"), and the code map is in [NAVIGATION.md](NAVIGATION.md).

The setup serves **Qwen3.8-27B** from one Intel Arc Pro B70 (32 GB) with the XPU backend of the TensorFold Zig engine,
a **GPTQ int4 group-128** target checkpoint, and the **DFlash2** drafter. Four pieces:

| Piece | Source | Format on disk | What the engine does with it |
|---|---|---|---|
| Target model | `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` (section 2), a quantization of `Qwen/Qwen3.8-27B` | GPTQ INT4, symmetric, group size 128, `desc_act=false`, MTP layer in BF16 | Reads the int4 words and fp16 scales as stored (`gptq-int4-g128`). |
| Tokenizer | the files shipped with that checkpoint (section 3) | `tokenizer.json`, `tokenizer_config.json` | Reads the pre-tokenizer from `tokenizer.json`. |
| Drafter | `z-lab/Qwen3.8-27B-DFlash2` (section 4) | BF16 | Quantized to affine int4 (groups of 64) **at load**. |
| Rotated weights (optional) | made from the base model and the target checkpoint (section 5) | GPTQ-shaped, rotated, with a `hadamard128` marker file | Faster prompts; **readable by this engine only**. |

## 1. What you need

- An Intel Arc Pro B70 (Battlemage G31, 32 GB) with the `xe` kernel driver and Intel's compute runtime (Level Zero).
  The engine opens the device by its PCI id.
- Zig 0.17, `ocloc` (Intel's offline compiler), a C++17 compiler, and the nGEN headers of oneDNN
  (`<ngen>`: the directory that holds `ngen.hpp`).
- For structured output: a checkout of [xgrammar](https://github.com/mlc-ai/xgrammar) at tag `v0.2.8`, with submodules.
- For making checkpoints: Python with `gptqmodel` **7.3.2**, `torch`, `safetensors`, and enough memory and a GPU for
  GPTQ calibration of a 27B model. `TODO (hardware and time used for our quantization run)`.
- Disk for the base weights, the quantized copy and the drafter.

Paths below use placeholders: `<models>` is a directory for checkpoints, `<repo>` a clone of this repository.

## 2. The target checkpoint: GPTQ INT4, group 128

We serve the checkpoint published by **SergiioB**, author of the Intel Arc Pro B70 inference cookbook, at revision
`9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e` (Apache-2.0, like the base model). Download it as published:

```sh
python -c "from huggingface_hub import snapshot_download as d; d('SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16', revision='9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e', local_dir='<models>/Qwen3.8-27B-GPTQ-g128')"
```

The rest of this section is how such a checkpoint is made, from its `quantize_config.json`, for anyone who wants to
redo it. Its author does not document the calibration set, so a redone checkpoint will not be bit-identical.

Start from the original bf16 checkpoint:

```sh
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3.8-27B', local_dir='<models>/Qwen3.8-27B')"
```

Quantize it with **gptqmodel 7.3.2** and these settings:

| Setting | Value |
|---|---|
| Bits / group size | 4 / 128 |
| Symmetric | yes (the zero point is fixed at 8: the engine never reads `qzeros`) |
| `desc_act` | `false` (no `g_idx`: the engine does not read one) |
| Layers kept unquantized | the MTP layer, kept in BF16, excluded with `dynamic: {"-:.*mtp.*": {}}` |
| Hessian damping `damp_percent` | 0.05 |
| `damp_auto_increment` | 0.01 |
| `true_sequential` | on |
| `act_group_aware` | on |
| Fallback | RTN, at a threshold of 0.5 % |
| Packing | int32 words |
| Calibration dataset | `TODO (calibration dataset to be confirmed)`: not recorded; number of samples and sequence length also `TODO` |

`TODO`: the exact key names of the fallback and of the packing in gptqmodel 7.3.2's `QuantizeConfig`. Take them from
the `quantize_config.json` of the checkpoint we published once it is confirmed, and keep that file with the checkpoint.
A sketch of the call, for the keys that are certain:

```python
from gptqmodel import GPTQModel, QuantizeConfig

cfg = QuantizeConfig(
    bits=4, group_size=128, sym=True, desc_act=False,
    damp_percent=0.05, damp_auto_increment=0.01,
    true_sequential=True, act_group_aware=True,
    dynamic={"-:.*mtp.*": {}},
    # TODO: fallback (RTN, 0.5 %) and int32 packing, with their gptqmodel 7.3.2 key names
)
model = GPTQModel.load("<models>/Qwen3.8-27B", cfg)
model.quantize(calibration)          # TODO (calibration dataset to be confirmed)
model.save("<models>/Qwen3.8-27B-GPTQ-g128")
```

### What the engine requires of the checkpoint

These are the loader's expectations (`zig/src/xpu/qwen27.zig`, `loadGptq`). A mismatch stops the load with an error
rather than serving wrong output.

- A sharded safetensors checkpoint with a `model.safetensors.index.json`.
- Every quantized projection as `<name>.qweight` (int32, `[K/8, N]`) and `<name>.scales` (fp16, `[K/128, N]`). The
  engine detects the format from the presence of `in_proj_qkv.qweight`.
- Stored in **fp16** (otherwise `UnexpectedDtype` or `UnexpectedShape`): `model.embed_tokens.weight`, `lm_head.weight`
  and the GDN gates `linear_attn.in_proj_a.weight` / `in_proj_b.weight`. The small tensors (norms, convolution, `A_log`,
  `dt_bias`) may be bf16 or fp16; fp16 RMSNorm weights are taken as stored with the `gamma − 1` convention (the engine
  adds 1). Our checkpoint has them in fp16 because gptqmodel loaded the model in float16 (its `config.json` dtype) and
  wrote every unquantized tensor in that dtype (checked: `embed_tokens`, `lm_head`, `in_proj_a`, `in_proj_b` are F16).
- The output head is requantized at load to affine int4 (groups of 64) with MLX's algorithm; the embedding stays fp16.
- The MTP tensors stay in the checkpoint but the XPU engine does not read them.

## 3. The tokenizer: keep the checkpoint's files

Serve with the `tokenizer.json` and `tokenizer_config.json` shipped with the checkpoint, so that anyone who repeats this
setup gets the same token ids and the same replies. The engine reads the pre-tokenizer from `tokenizer.json`.

Known difference: these files were re-saved with `transformers` 5.14.1, whose pre-tokenizer regular expression lacks
`\p{M}`. Hindi, Thai and text with combining marks tokenize differently from the official `Qwen/Qwen3.8-27B` files
(for example `สวัสดีครับ`: 7 tokens instead of 3); plain French, English and code tokenize the same. Copying the official
files over them is possible, but then the token ids, and the replies on such text, differ from this recipe's.

## 4. The drafter: DFlash2

```sh
python -c "from huggingface_hub import snapshot_download as d; d('z-lab/Qwen3.8-27B-DFlash2', local_dir='<models>/Qwen3.8-27B-DFlash2')"
```

Use the BF16 weights as published. The directory must contain `model.safetensors`, including the
`candidate_selector` codebooks. The engine quantizes the projections to affine int4 (groups of 64) on the device at
load, whatever the target's format; there is no separate quantization step.

This is the repository the other backends of this project use (`zig/src/families/qwen27/dflash/config.zig`); its model
card states that it mirrors `incoai/Qwen3.8-27B-DFlash2`.

Drafts only change how many tokens are accepted: the target verifies every token, so the drafter cannot change a reply.

## 5. Optional: rotated weights

[`zig/tools/xpu/rotate_w4.py`](../../zig/tools/xpu/rotate_w4.py) remakes the quantized projections of the GPTQ checkpoint
from the original bf16 weights: each weight is multiplied by a normalized Hadamard matrix of size 128 along K, then
rounded to nearest (RTN: scale `2·max|w| / 15` per group of 128, zero point 8). The GDN gates `a` / `b` are rotated too.
Other tensors (embedding, norms, convolution, head, MTP, vision) are copied from the GPTQ checkpoint, and a small marker
file named `hadamard128` is written in the output directory. The engine then rotates every GEMM input where it is
produced.

```sh
python <repo>/zig/tools/xpu/rotate_w4.py \
    <models>/Qwen3.8-27B <models>/Qwen3.8-27B-GPTQ-g128 <models>/Qwen3.8-27B-rot rot
```

(The last argument is `rot` or `norot`; `norot` redoes the RTN quantization without the rotation, for comparison.)
The script copies the GPTQ directory's non-weight files, tokenizer included. Needs
`torch` and `safetensors`.

> **Warning: rotated checkpoints are readable by this engine only.** The weights are not the model's weights: they are
> the weights times a Hadamard matrix, and only this engine rotates the activations to match. Any other engine (vLLM,
> llama.cpp, `transformers`, another build of TensorFold) will load the files without complaint and **produce wrong
> output silently**. Do not publish a rotated checkpoint under the name of the model, label it clearly, and keep the
> `hadamard128` marker with it: without the marker this engine would also read the weights as ordinary ones.

The W4A4 prompt path (`XPU_PREFILL_INT4=1`, together with `XPU_PREFILL_INT8=1`) requires rotated weights and refuses to
start without them (`Int4NeedsRotatedWeightsAndInt8`). It and the W4A8 option are local options: see
[README.md](README.md) for what they trade.

## 6. Build and serve

```sh
cd <repo>
zig build xpu-native -Doptimize=ReleaseFast -Dngen-include=<ngen>   # zig-out/xpu/bin/tensorfold-native
zig build xpu-qwen27 -Doptimize=ReleaseFast -Dngen-include=<ngen>   # test executable xpu-qwen27
```

`-Dxpu-device=` (default `bmg`) and `-Docloc=` (default `ocloc`) are the other options. The kernels are compiled for the
device at build time and embedded in the executables; see [NAVIGATION.md](NAVIGATION.md#2-building).

Serve:

```sh
<repo>/zig-out/xpu/bin/tensorfold-native serve <models>/Qwen3.8-27B-GPTQ-g128 \
    --drafter <models>/Qwen3.8-27B-DFlash2 --parallel 8 --context 16384 --host 127.0.0.1 --port 8081
```

- `--parallel N` gives each stream a slot on the card (a GDN state and a K/V cache of `--context` tokens). The number of
  slots is capped by the memory left after loading; the server logs the number of streams it got.
- With one active stream the engine drafts DFlash2 trees; with several it shares rounds between them, for any
  `--parallel` value.
- Environment options (`XPU_PREFILL_INT8`, `XPU_KV_INT8`, `XPU_TREE`, …) are listed in [README.md](README.md) and
  [NAVIGATION.md](NAVIGATION.md#9-environment-variables). The defaults are the validated configuration; the others are
  options.

### Structured output

`response_format` (JSON, `json_schema`) and the `guided_*` fields need the grammar library, built from xgrammar:

```sh
git clone --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar <xgrammar>
cd <repo> && zig build grammar -Dxgrammar=<xgrammar>
```

The build checks that the checkout is at tag `v0.2.8` and produces `libtfgrammar.so`. The server looks for it in
`<server directory>/../lib/`, then next to the server, then on the loader's path, and loads it at the first structured
request: put it in `zig-out/xpu/lib/` (`zig build grammar` installs under `<prefix>/lib`; pass `--prefix zig-out/xpu` or
copy the file; `zig build grammar` then a copy of `zig-out/lib/libtfgrammar.so` into `zig-out/xpu/lib/` is what we
run). Without the library, structured requests are
refused and everything else works. If you redistribute the library, ship xgrammar's licence and notice with it
(Apache-2.0).

## 7. Validate

With real prompts, not synthetic ones. `ids.txt` holds the token ids of a prompt, separated by spaces, produced with the
checkpoint's tokenizer (for example with the `tokenizers` package: `Tokenizer.from_file("<models>/Qwen3.8-27B-GPTQ-g128/tokenizer.json")
.encode(text).ids`).

```sh
cd <repo>
zig build test

M=<models>/Qwen3.8-27B-GPTQ-g128
D=<models>/Qwen3.8-27B-DFlash2
X=zig-out/bin/xpu-qwen27

$X $M ids.txt 32                                   # 32 greedy tokens: compare with the fp32 torch reference
XPU_VERIFY_TEST=1 $X $M ids.txt 32                 # drafted == serial, perfect and wrong drafts
XPU_DRAFTER=$D XPU_TREE=15 $X $M ids.txt 64        # DFlash2 trees == serial
XPU_MULTI_TEST=8 $X $M ids.txt 32                  # 8 streams in shared rounds == serial (chains and trees)
```

- **Oracle.** The greedy tokens must equal the fp32 torch reference of the Python port for the same prompt (32 tokens).
  The reference script lives with the Python port's bench tools and is not published yet. `TODO (publish it, or ship a
  recorded token list per prompt)`.
- **Drafted == serial, multi-stream == serial.** The runner prints `equal to serial: true` for each check; any `false`
  is a bug, not noise.
- **Server level**, against a running server:

```sh
python3 tools/bench_concurrent.py http://127.0.0.1:8081 <model name> --levels 1,2,4,8 --alone --serial
python3 tools/bench_openai.py http://127.0.0.1:8081 <model name>
```

`--alone` checks that every concurrent reply is token-for-token the reply of the same request alone, and `--serial`
checks the alone runs against `"draft": false`. Both greedy and sampled requests are covered by default. Run it before
and after any change to a kernel or to the quantization.

If you made rotated weights, run the same checks on that directory, and compare perplexity on raw text against the GPTQ
checkpoint with the studies in [`tools/xpu/`](../../tools/xpu). Do not measure on chat-formatted prompts: in a user turn
the model predicts the next token badly, so perplexities there mean nothing.

## 8. Results

This page does not repeat numbers. Prompt rates, single-stream and multi-stream decode rates, and the validation that
was run (oracle, drafted == serial, multi-stream == serial, the 64-cell concurrent check) are in
[README.md](README.md), sections "Numbers" and "Correctness: what is verified", with the conditions they were taken in
(power limit, `--parallel`, `--context`, text). Rates depend on the card's power limit and on the text produced; record
the engine commit, the checkpoint and drafter revisions, and the power limit with any figure you report.

## 9. Licences and attribution

The base model and the drafter are both released under the **Apache License 2.0**. A quantized or rotated checkpoint
is a modified copy of the base weights, so if you redistribute it:

- include a copy of the Apache-2.0 licence text;
- keep the upstream copyright, attribution and trademark notices, and include the contents of the upstream `NOTICE`
  file if there is one (`TODO: check the upstream repositories for a NOTICE file`);
- credit the quantized checkpoint's author: the GPTQ checkpoint is SergiioB's
  (`SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`);
- state that you changed the files: for example "GPTQ int4 (group 128, symmetric) quantization of `Qwen/Qwen3.8-27B`
  made with gptqmodel 7.3.2; MTP layer kept in bf16", and, for rotated weights, "weights multiplied by a Hadamard
  matrix of size 128 and rounded to nearest; readable only by TensorFold's XPU engine";
- do not use the upstream names or marks to suggest that the upstream authors endorse your files.

The drafter is quantized at load and never rewritten, so serving it as published needs nothing beyond keeping its
licence with the download. TensorFold itself is Apache-2.0 (see [LICENSE](../../LICENSE) and [NOTICE](../../NOTICE)); the
xgrammar library you may build is Apache-2.0 too, with its own notice. This section is a reminder, not legal advice:
read the licence text of each upstream repository.
