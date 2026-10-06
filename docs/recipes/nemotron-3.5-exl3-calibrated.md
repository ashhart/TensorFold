# Reproduce Nemotron 3.5 Lightning BF16 → calibrated EXL3 → TensorFold CUDA

This is the exact local route used to serve a self-calibrated EXL3 conversion of
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
on one NVIDIA DGX Spark. It does **not** substitute TensorFold's separate MLX-4bit
checkpoint, alter NVIDIA's BF16 weights, or download a published EXL3 checkpoint.
The quantized weights have **not** been uploaded; run the conversion below to
recreate them. The patch is in TensorFold's `feat/nemotron-exl3-native` branch
initially published by NeoAiLabs (and its upstream PR), not upstream `main`
unless that PR has merged.

## Pinned inputs and tested environment

- BF16 source revision: `a9904d24bcc1d289a1950fa9d2b978c47cf903b9`
  (`NemotronHForCausalLM`, 14 indexed safetensors shards).
- ExLlamaV3: `v1.5.3`, commit `d3739fd393337b1ff4d6c2a342b12f0c87a9592f`.
- TensorFold base: `609ca419abecebdc5a059498a613680bd3aa847f`
  (0.6.5), plus the native Nemotron EXL3 implementation in this branch.
- Tested host: aarch64 DGX Spark / GB10 SM121, Ubuntu 24.04.5, CUDA toolkit
  13.0.88; isolated Python 3.12.3, PyTorch `2.13.0+cu130`, Triton 3.7.1,
  safetensors 0.8.0, tokenizers 0.23.1. `nvcc` and the PyTorch CUDA build must
  be compatible with SM121. Keep the existing NVIDIA driver/CUDA OS stack.
- The tested ExLlamaV3 virtual environment already existed; this is **not** a
  claim that a fresh `pip install torch==2.13.0+cu130` works on every host.
  Follow ExLlamaV3's pinned installation instructions for your GPU platform,
  then use its environment for these commands. The TensorFold server was run
  from source with `PYTHONPATH=src` in that environment, not from a container.
- Allow ample NVMe space for ~66 GB of BF16 shards, the ~18 GB EXL3 output,
  a resumable conversion work directory, and safety margin. Check capacity
  before starting. Conversion/calibration can take hours.

## 1. Obtain and verify the unquantized source

Run from a shell with `hf` installed and access to the source model:

```bash
MODEL_ID=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
REV=a9904d24bcc1d289a1950fa9d2b978c47cf903b9
ROOT="$HOME/exl3-nemotron-35"
BF16="$ROOT/bf16"
VIEW="$ROOT/source-view"
CAL="$ROOT/calibration"
OUT="$ROOT/quants/4.00bpw-hq-cal"
WORK="$ROOT/work/4.00bpw-hq-cal"
EXLLAMA="$HOME/ai/exllamav3"
TF="$HOME/Projects/TensorFold"
PY="$EXLLAMA/.venv/bin/python"
mkdir -p "$ROOT" "$CAL"
df -h "$ROOT"
HF_HUB_DOWNLOAD_TIMEOUT=60 HF_XET_NUM_CONCURRENT_RANGE_GETS=4 \
  hf download "$MODEL_ID" --revision "$REV" --local-dir "$BF16" --max-workers 2
hf cache verify "$MODEL_ID" --revision "$REV" --local-dir "$BF16" --fail-on-missing-files
```

The original source `config.json` SHA-256 in this run was
`a3827a0f5e311547b40943dc081e3ff2f8a277466e8c1a3df2291e8db8a7617c`.
Do not replace it with a quantized/NVFP4 variant or edit it to satisfy a loader.

## 2. Pin the conversion and TensorFold implementations

```bash
git clone https://github.com/turboderp-org/exllamav3.git "$EXLLAMA"
git -C "$EXLLAMA" checkout d3739fd393337b1ff4d6c2a342b12f0c87a9592f
# Install/build this pinned checkout in an isolated SM121-capable CUDA environment
# following that checkout's instructions; point PY at its Python executable.
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())'

git clone https://github.com/NeoAiLabs/TensorFold.git "$TF"
git -C "$TF" checkout feat/nemotron-exl3-native
```

The `EXLLAMA` checkout and `PY` are the actual paths used for conversion here,
expressed relative to the reader's home directory. Do not overwrite an
existing checkout or environment with these `git clone` commands; reuse it
only after verifying the pinned commit and compatible package versions.
The conversion and server do not require the evaluation-only local qbench patch
recorded in the quant manifest.

## 3. Prepare a non-mutating source view

At this pinned BF16 revision NVIDIA's config calls trunk blocks `mamba`,
`attention`, and `moe`, whereas ExLlamaV3 v1.5.3 expects explicit
`hybrid_override_pattern` and `mtp_hybrid_override_pattern`. The helper
[`tools/prepare_nemotron_exl3_source.py`](../../tools/prepare_nemotron_exl3_source.py)
checks the source architecture, layout, index, and 14 shards. It writes **only**
a new config under `VIEW` and symlinks other files back to `BF16`; it also
maps block-name aliases for current Transformers tokenizer configuration.

```bash
"$PY" "$TF/tools/prepare_nemotron_exl3_source.py" \
  --source "$BF16" --overlay "$VIEW" --require-weights
```

The adapted view's config SHA-256 was
`f3607fcfb547ffa6b2c914e90db0e2ebc326c3d4f113168625a3587a1185b094`.
If the source revision or config changes, inspect rather than forcing this hash
or modifying the BF16 files.

## 4. Generate self-calibration and convert

The BF16 source view, not the low-bitrate target, generated the calibration.
The executed commands below come from the conversion logs (shell variables
replace machine-specific absolute paths). This was ExLlamaV3's HQ 4.00-bpw
request with a 6-bit head, its `mul1` codebook and `apply_out_scales` defaults;
there was no optimized per-tensor recipe. Keep the work directory for resume.

```bash
cd "$EXLLAMA"
"$PY" sc_trace.py -m "$VIEW" -cs 32768 -ambs 8 \
  -o "$CAL/cal_trace.json" -co "$CAL/cal_trace.safetensors" \
  -cr 128 -cc 2048 --target_tokens 262144 --seed 5151 \
  --max_new_tokens 3072 --max_epochs 4

"$PY" convert.py -i "$VIEW" -o "$OUT" -w "$WORK" \
  -b 4.00 -hb 6 -hq -cd "$CAL/cal_trace.safetensors" -d 0
```

For this run, the trace reported 157 generated rows and the converter consumed
250 packed rows × 2048 columns. The converter produced three indexed weight
shards totaling 18,065,353,915 bytes; the measured EXL3 layer/head rates were
4.1107/6.0061 bpw, not uniformly 4.00 bits per weight. Keep `config.json`,
`generation_config.json`, tokenizer files, and the index alongside all shards.
Do not use this conversion step on a checkpoint that is already quantized.

If only serving is needed and you already have a compatible local Nemotron-H
EXL3 checkpoint, skip sections 1–4, but verify its exact EXL3 group layout;
this implementation has been exercised on the calibrated output above.

## 5. What changed in TensorFold

This was a native loader/kernel change, **not** a metadata edit to fool `info`:

1. `families/nemotron_h/__init__.py` admits EXL3 only on CUDA and refuses MTP
   drafting and two-rank tensor parallelism for this format.
2. `families/nemotron_h/cuda/exl3_weights.py` reads packed groups and plain
   tensors across safetensors shards, trims padded projections to logical
   dimensions, concatenates separate Q/K/V in order, and checks for unconsumed
   trunk tensors. `weights.py` dispatches to that loader. The original quant
   shards are unchanged.
3. `cuda/exl3/experts.py`, `experts.cpp`, and `experts.cu` add single-GPU,
   gateless ReLU² routed experts (one up/down pair per expert), zero padded
   intermediate channels **before** the down projection, skip non-routed IDs,
   and leave the existing generic SwiGLU path intact. The shared expert uses
   separate EXL3 up/down projections and is added in `engine.py`.
4. `glue.py` and `engine.py` run both prefill and decode using these EXL3
   projections and a plain embedding tensor. `Config.read` merges EOS IDs from
   `config.json` and `generation_config.json`: here the chat end token
   `<|im_end|>` is ID 11. Without this fix the server returned HTTP 200 but
   repeated the answer and special markers until `max_tokens`.

## 6. Check loading and inference before exposing the endpoint

```bash
cd "$TF"
TENSORFOLD_NEMOTRON_EXL3_MODEL="$OUT" PYTHONPATH=src "$PY" -m pytest \
  tests/test_nemotron_exl3_source_view.py \
  tests/cuda/test_nemotron_exl3.py \
  tests/cuda/test_nemotron_exl3_experts.py -q
PYTHONPATH=src "$PY" -m tensorfold info "$OUT"
```

The focused tests include a real calibrated-checkpoint load, prefill, decode,
synthetic and real expert differential checks, CUDA graph replay, EOS behavior,
and explicit unsupported-mode rejections. Three tests requiring an actual
checkpoint skip unless `TENSORFOLD_NEMOTRON_EXL3_MODEL` is set. This does not prove
all possible Nemotron quants or a broad quality benchmark. Three unrelated
existing EXL3 tests in this environment call an installed ExLlamaV3
`reconstruct` binding with float half-bit widths where it requires an integer;
they are not evidence about this Nemotron loader.

Start serial on one GPU, using the same local output. Restrict to loopback;
set a private API key file before allowing tailnet clients:

```bash
# A new key file, mode 0600 (do not print it or commit it).
KEY_FILE="$HOME/.config/tensorfold/nemotron-exl3.keys"
python3 - "$KEY_FILE" <<'PY'
import os, pathlib, secrets, sys
p = pathlib.Path(sys.argv[1])
p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as out:
    out.write(secrets.token_urlsafe(32) + '\n')
PY

cd "$TF"
PYTHONPATH=src "$PY" -m tensorfold serve "$OUT" \
  --backend cuda --tp 1 --no-drafts --context 8192 --parallel 1 \
  --host 127.0.0.1 --port 8080 --name nemotron-3.5-lightning-exl3 \
  --api-key-file "$KEY_FILE" --no-update-check
```

`--context 8192` is the requested prompt/reply window; inspect the capacity
line at startup (this run allocated 8208 cache slots and reported a rounded
8704-token engine context). The run is single-stream, no speculative drafting,
BF16 prompt activations. The foreground command above is the portable recipe;
our live instance was started detached with the same flags.

Verify authentication, content, and EOS; HTTP readiness alone is insufficient:

```bash
API_KEY="$(< "$KEY_FILE")"
curl -fsS -H "Authorization: Bearer $API_KEY" http://127.0.0.1:8080/v1/models
curl -fsS -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  http://127.0.0.1:8080/v1/chat/completions \
  -d '{"model":"nemotron-3.5-lightning-exl3","messages":[{"role":"user","content":"What is the capital of France? Reply with just the city."}],"temperature":0,"max_tokens":64,"stream":false,"chat_template_kwargs":{"enable_thinking":false}}'
```

The checked response contained `Paris` with `finish_reason: stop`, not repeated
`<|im_end|>` markers. A second one-word request and a one-sentence mutex
explanation also stopped cleanly; the default thinking template returned a
separate reasoning field and `Paris` in content. Requests without the key get
HTTP 401. Keep the backend on loopback; use Tailscale Serve when authorized,
or a proxy bound **only** to the machine's Tailscale IP (not `0.0.0.0`) for
remote access. The latter was used on this host because Tailscale Serve was
not enabled on its tailnet. A local probe to that IP does not verify another
node's ACL.

## Measured scope and limitations

On the DGX Spark, the calibrated EXL3 quant passed ExLlamaV3's seven-case
smoke suite and a disjoint held-out qbench (21,204 response tokens, BF16
reference PPL 1.134945, calibrated PPL 1.170566, mean KLD 0.039908).
Those are **ExLlamaV3 quantization** measurements, not TensorFold quality
scores. The TensorFold live server's bounded single-stream HTTP probe used
one warm-up and three timed, cold-prompt 128-token responses at each length:

- ~81 prompt tokens: median first content token 0.111 s; decode 87.02 tok/s.
- ~506 prompt tokens: 0.274 s; 86.62 tok/s.
- ~2050 prompt tokens: 0.947 s; 86.51 tok/s.

No prefix cache was used. Time to first content token includes HTTP,
rendering, prefill, and first-token work; it is **not** pure prefill throughput.
This probe is not a controlled comparison to the ExLlamaV3 offline speed test.
Memory allocation observed in the live server was about 19 GiB on this host.
Do not treat loadability or these small smoke/throughput tests as full quality,
concurrency, extended-context, or MTP validation. EXL3 MTP and TP=2 are
explicitly unsupported by this implementation. No model weights, Hugging Face
model card, API keys, private IP addresses, or host-specific filesystem paths
are included in this recipe or the code patch.
