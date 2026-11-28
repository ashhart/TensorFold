# Reproduction commands

Use the source commits and file hashes recorded in `summary.json`, the pinned
public snapshots in `models.json`, Zig 0.17.0 and NVCC 13.0.88. Use Python
3.11 or newer for the clients; the full CPU reference uses the pinned dependency
versions in the summary. Commands below
use generic local paths. Run one model process under the existing GPU lock on
an authorized test machine. Preserve unrelated deployments and local work.

## Build and contracts

```sh
zig build fatbins kernel-contract-fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=80,86,89,120,121
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=80,86,89,120,121 -Doptimize=ReleaseFast -Dfatbins=zig-out/fatbin -Dcontract-controls=zig-out/fatbin/contract_controls.fatbin
zig build native qwen35-score-reference -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=80,86,89,120,121 -Doptimize=ReleaseFast -Dfatbins=zig-out/fatbin -Dcontract-controls=zig-out/fatbin/contract_controls.fatbin
zig build test test-kernel-contract test-golden fuzz-cortheon -j8 --summary all
zig-out/bin/tf-cuda-test kernel-contract-limits zig-out/fatbin
flock -n /tmp/tensorfold-gpu.lock zig-out/bin/tf-cuda-test kernel-contract
python3 tools/zig/lean_check.py
python3 tools/zig/lean_new.py BASE.txt HEAD.txt
python3 -m unittest discover -s tools/zig -p 'test_qwen35_reference*.py'
python3 -m unittest discover -s tools/zig -p test_qwen35_served_reference.py
```

`kernel-contract-limits` is a file-only audit; the complete `kernel-contract`
command additionally runs numerical gates, controls and live device checks.
The `qwen-f32-norm-contract` diagnostic runs the two new norm entries alone;
it is not the complete backend receipt. Lean uses the same checker on both
trees and requires zero new findings; inherited findings are counted.

The original plain/native builds omitted `-Dsm` for its independently compiled H3
image. Exact original container argv are archived in `host/build-commands.json`.
The first full CUDA attempt therefore failed with `CUDA_ERROR_NO_BINARY_FOR_GPU`
before numerical gates. The explicit five-SM rebuild above corrects that build
setup. Its runner has a separate hash; all 141 common file images were compared
and only the H3 embedding changes. The served native binary stays byte-identical.
The failed attempt, corrected build log and byte-level proof are in the asset.

## Independent FP64 and Transformers

```sh
flock -n /tmp/tensorfold-gpu.lock zig-out/bin/tf-qwen35-score-reference MODEL docs/receipts/qwen35-bf16-scoring/reference-fixtures.json captures
python3 tools/zig/qwen35_scores_fp64.py MODEL captures --revision fc05daec18b0a78c049392ed2e771dde82bdf654 --output fp64.json
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 python3 tools/zig/qwen35_transformers_reference.py --stage all --model MODEL --fixtures docs/receipts/qwen35-bf16-scoring/fixtures.json --api API.json --fp64 FP64.json --captures CAPTURES --revision-record REVISION.json --fp32-4097 --output transformers.json
python3 tools/zig/check_qwen35_served_reference.py http://127.0.0.1:8091 qwen35-bf16 docs/receipts/qwen35-bf16-scoring/fixtures.json transformers.json --output served-reference.json
```

For byte-for-byte input provenance, the corrected CPU experiment uses the
historical API continuations/capture tree whose hashes are in its `input_files`.
The actual HTTP comparison uses the current server. Its independent BF16/FP32
prompt rows are causal and do not use the native auxiliary capture results as
an oracle. Preserve `complete`, strict diagnostic status and exit1 separately;
a diagnostic failure does not mean the forward was omitted or truncated.
Do not remove original FP32 parameter restoration or skip the 4097 FP32 arm.

## Native server and served clients

Serving was containerized with four CPUs, 32 GiB RAM, one GPU and a loopback host
port. Adjust paths inside an otherwise equivalent container; the listener may
bind its private bridge interface while only 127.0.0.1 is published to the host.

```sh
flock -n /tmp/tensorfold-gpu.lock zig-out/native/bin/tensorfold-native serve MODEL --name qwen35-bf16 --host 127.0.0.1 --port 8091 --context 8192 --parallel 4 --prompt-cache-gib 0 --no-thinking --no-update-check --dashboard
python3 tools/zig/check_qwen35_prompt_scores.py http://127.0.0.1:8091 qwen35-bf16 docs/receipts/qwen35-bf16-scoring/fixtures.json --output api.json
python3 tools/bench_openai.py http://127.0.0.1:8091 qwen35-bf16 --tokens 128 --reps 3 --temperatures 0 --output decode.json
python3 tools/bench_concurrent.py http://127.0.0.1:8091 qwen35-bf16 --tokens 128 --reps 3 --temperatures 0,0.7 --levels 1,4 --alone --serial --output concurrent.json
python3 docs/receipts/qwen35-bf16-scoring/reproduce-cache.py http://127.0.0.1:8091 qwen35-bf16 CHAT-FIXTURES.json fresh --output fresh.json
```

Restart with `--prompt-cache-gib 2`, then run retained mode with
`--reference fresh.json`. Restart again with cache0, repeat fresh mode and
compare every reply hash against the first fresh run. Cancellation uses:

```sh
python3 docs/receipts/qwen35-bf16-scoring/reproduce-disconnect.py http://127.0.0.1:8091 qwen35-bf16 docs/receipts/qwen35-bf16-scoring/fixtures.json --output cancellation.json
```

For each cold length 2048/8192/32768/65536, start a fresh cache0/C1 server at
context N+32 and run `reproduce-cold.py SOURCE COLD-FIXTURE.json OUTPUT.json`.
The archived fixture includes exact IDs and the rendered request item.
Memory sampling is a separate untimed run: cache0/context8192/C4,
cache2 during retained exactness, and cache0/context65568/C1 admission.
The memory admission run sends 4097 IDs unscored/scored with eight reply tokens.
Record runtime device/pinned/RSS counters, `/proc` VmHWM and cgroup memory.peak
without equating their scopes or publishing process identifiers.

## Existing formats and SGLang

Use pinned-main and candidate native binaries against the same prebuilt
assets, checkpoint and quotas, in baseline/candidate/baseline/candidate order.
For each MLX4 and ModelOpt process, run the decode command above; run concurrency
with one repetition in the first two processes. Request `prompt_logprobs:2`
with `[1,2,3]` and one reply token from every process and retain its actual
HTTP 400 refusal. Compare every baseline/candidate full reply hash. Retain the
selected EXL3 export's actual startup outcome and do not claim inference after
`UnexpectedTensor`.

SGLang image:
`lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9`.
Mount the attached `sglang-precision` folder at `/policy` and set
`PYTHONPATH=/policy`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
Launch `python3 -m sglang.launch_server` under the same GPU lock and quotas:

```sh
--model-path MODEL --served-model-name qwen35-bf16 --host 0.0.0.0 --port 8091 --dtype bfloat16 --kv-cache-dtype bfloat16 --mamba-ssm-dtype float32 --context-length 8192 --max-running-requests 4 --max-total-tokens 32768 --chunked-prefill-size 4096 --tp-size 1 --mem-fraction-static 0.85 --attention-backend triton --disable-radix-cache --mamba-radix-cache-strategy no_buffer --enable-cache-report --cuda-graph-max-bs-decode 4
```

Use the same decode/concurrency clients, omitting native `--alone --serial`
checks because SGLang has no native token hash. For each cold length use a fresh
process, context N+32, C1, total tokens N+32 and graph batch 1. Keep the generic
cold client's null cache counters and failed qualifier; separately retain the
scheduler's observed cached/new-token accounting. No missing counter is zero.
