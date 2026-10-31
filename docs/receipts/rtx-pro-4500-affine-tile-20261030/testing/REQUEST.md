## Testing request: RTX 5090 and RTX PRO 6000 Blackwell owners

Could someone with either card test this improvement independently? Please specify the PRO 6000 edition (Workstation, Max-Q or Server) and power limit. **Use only PR #1730; do not combine it with the companion PR.**

A startup-only experimental override is now included:

| Setting | Effect |
|---|---|
| `TF_CUDA_TEST_MLP128=0` | Original kernel path: baseline |
| `TF_CUDA_TEST_MLP128=1` | Candidate path on SM120, including other SM counts |
| Unset | Existing automatic policy: candidate only on SM120/82 SMs |

Other architectures cannot be forced on. Model/shape, precision, KV, context and graph guards remain in place. For attention, wider windows are still selected only at 8192<=cached<65536; the override does not force every attention shape. The flags are read at startup, so **restart the server between arms**. Exporting the flag only in the benchmark client does nothing.

Commands below assume Linux/Bash.

### 1. Isolated checkout and one build

```sh
git clone --filter=blob:none --no-checkout https://github.com/ashhart/TensorFold.git tensorfold-pr1730-test
cd tensorfold-pr1730-test
git fetch origin pull/1730/head
git checkout --detach FETCH_HEAD
git rev-parse HEAD
zig build native -Dsm=120 -Dnvcc="$(command -v nvcc)" -Dversion=pr1730-test -j4
```

Use Zig 0.17.0 and preferably the same CUDA 13.0 or 13.3 toolchain for all arms; record full versions. Reuse your qualified grammar/media library setup. Use the same `TensorFold/Qwen3.8-27B-MLX-4bit` revision `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c` and BF16 `z-lab/Qwen3.8-27B-DFlash2` revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` throughout.

### 2. Server terminal: baseline, candidate, baseline again

Set MODEL_DIR and DRAFTER_DIR to your existing checkpoint directories. Stop other GPU workloads during each measured arm. Start with 0, then repeat with 1, then 0 again. Set ARM respectively to `baseline`, `candidate`, `baseline-repeat`; each invocation gets fresh calibration storage.

```sh
unset TF_CUDA_TEST_MLP128 TF_CUDA_TEST_ATTN_MT2
MODEL_DIR=/path/to/Qwen3.8-27B-MLX-4bit
DRAFTER_DIR=/path/to/Qwen3.8-27B-DFlash2
ARM=baseline
TF_CUDA_TEST_MLP128=0 TENSORFOLD_MEMORY_RESERVE_GIB=0.5 HOME="$(mktemp -d)"   ./zig-out/native/bin/tensorfold-native serve "$MODEL_DIR"   --drafter "$DRAFTER_DIR" --name bench --host 127.0.0.1 --port 18080   --context 131072 --parallel 1 --kv-cache fp8 --prompt-precision int8   --vision --prompt-cache-gib 8 --max-tokens 2048   >"$ARM-server.log" 2>&1
```

Confirm the log says `experimental MLP128 override=0, enabled=false` for baseline and `override=1, enabled=true` for candidate. An enabled device flag permits the candidate only on its eligible shapes; it does not mean every launch uses it. Keep the same power settings, context, vision, caches and precision in all arms. If this configuration cannot fit, report that rather than silently reducing features in one arm.

### 3. Client terminal: identical sequence for every arm

Wait until `/health` is ready and warming is false. Set ARM to match the server terminal, then run:

```sh
ARM=baseline
RECEIPT=docs/receipts/rtx-pro-4500-affine-tile-20261030
python3 tools/bench_openai.py http://127.0.0.1:18080 bench   --tokens 256 --reps 5 --temperatures 0,1 --output "$ARM-short.json"
python3 "$RECEIPT/testing/replay_long.py" http://127.0.0.1:18080   "$RECEIPT/testing/fixture-8192.json" --model bench --repetitions 3   --output "$ARM-long8k.json"
python3 "$RECEIPT/testing/replay_long.py" http://127.0.0.1:18080   "$RECEIPT/testing/fixture-32768.json" --model bench --repetitions 3   --output "$ARM-long32k.json"
```

The long replay runs both `draft=true` and `draft=false`, checks repeated hashes, and asserts drafted/plain token equality. After all three arms, also check baseline/candidate hashes:

```sh
python3 - <<'CHECK'
import json
for size in ['8k', '32k']:
    reference = None
    for arm in ['baseline', 'candidate', 'baseline-repeat']:
        samples = json.load(open(f'{arm}-long{size}.json'))['samples']
        hashes = {(x['drafted'], x['repeat']): x['response']['tensorfold']['token_sha256'] for x in samples}
        if reference is None:
            reference = hashes
        assert hashes == reference, (size, arm, 'token mismatch')
print('Long-fixture hashes match across all three arms.')
CHECK
```

### Please attach

GPU model/edition and power limit; driver, nvcc, Zig, OS and Git SHA; checkpoint revisions and unchanged serving settings; all nine benchmark JSON files and the three server logs. Include the short benchmark's `evidence` hashes, not just headline tok/s. Report short, 8K, 32K, plain and drafted results separately, including regressions. For the long runs use median server decode tok/s; compare candidate against the geometric mean of the two baseline medians. Report prefill/TTFT separately, and keep cached-token counts visible.

This is a focused performance/exactness request, not a request to run an hours-long full suite. The PR remains draft: full backend contract SUMMARY qualification and the other documented release gates are still outstanding. No GPU run of this newly added override has been performed locally; pure override tests and targeted host checks passed. The original 4500 receipts remain pinned to their original measured commits. Please report failures without disabling hash checks or changing precision.
