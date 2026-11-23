# Dense MTP receipts for issue 2485

These scripts capture reproducible same-GPU receipts for the dense MTP change. Reports and logs go under `zig-out/bench` by default. The performance gate uses the uninstrumented first request from a fresh server; diagnostic profiling is intentionally excluded.

All runners accept `--project`, `--base-worktree`, `--checkpoint`, and `--output-dir`. The full serving receipt also accepts `--nvcc` and `--model`. Equivalent paths can be set with `TF2485_PROJECT`, `TF2485_BASE_WORKTREE`, `TF2485_CHECKPOINT`, `TF2485_OUTPUT_DIR`, `TF2485_NVCC`, and `TF2485_MODEL`.

## Full serving and memory receipt

Builds both native binaries, then records target-only and MTP decoding, cold prompt latency, token hashes, GPU memory, and supported stream levels:

```sh
python3 tools/bench/mtp2485/run_receipt.py \
  --base-worktree /path/to/main-worktree \
  --checkpoint /path/to/target-checkpoint \
  --nvcc /path/to/nvcc
```

The server follows an explicit lower `--parallel` recommendation after a memory refusal, otherwise backing off one stream at a time. Any unadmitted stream level is recorded as a capacity skip, never as a measurement.

## Cold prompt performance gate

Runs paired fresh-server samples at 2,048 and 8,192 prompt tokens by default. It compares main, candidate without drafts, candidate with MTP, and candidate with DFlash2. The performance gate allows up to 0.1% median TTFT regression versus main as run-to-run noise; target-only and MTP output hashes must match main, and every required sample must be cold. Candidate-no-drafts and MTP-versus-no-drafts deltas are reported to isolate overhead, but are diagnostic rather than independent gates. DFlash2 is also a diagnostic reference, not a speed gate; its cold timing and output hash are recorded beside MTP when it fits. A DFlash2 memory-capacity refusal is recorded as capacity-limited and does not fail the main-versus-branch gate:

```sh
python3 tools/bench/mtp2485/prompt_ordered.py \
  --base-worktree /path/to/main-worktree \
  --checkpoint /path/to/target-checkpoint \
  --dflash2-drafter /path/to/dflash2-checkpoint \
  --reps 5
```

The DFlash2 checkpoint is required for this gate and must be a local directory accepted by the same target. Its path and checkpoint metadata hashes are recorded in the report. Set `TF2485_DFLASH2_DRAFTER` instead of the flag when preferred.

Optional longer prompts can be added with `--prompt-tokens 2048,8192,32768,65536` where main supports those contexts. Unsupported contexts are reported as skips and cannot satisfy either required gate context. Keep `TF_TIMING` unset for these measurements.

The direct-normalized MTP prompt path retains 256 ordinary prompt rows for its initial draft state and handles the final row separately. Other paths keep the full tap window. The 256-row setting is fixed; it has no runtime override.

## Greedy, seeded sampling, and resume correctness

The OpenAI receipt compares main / MTP chain / MTP sampled tree / main at one stream, with two prompts at greedy and seeded temperature 1.0 and three valid output hashes per cell. `TF_MTP_TREES=1` is set only for the sampled-tree run; chain mode remains the default:

```sh
python3 tools/bench/mtp2485/bench_openai_ordered.py \
  --base-worktree /path/to/main-worktree \
  --checkpoint /path/to/target-checkpoint
```

The resume receipt compares a canonical target-only history against candidate no-draft and MTP continuation, requiring at least 4,096 cached prompt tokens:

```sh
python3 tools/bench/mtp2485/resume_exactness.py \
  --base-worktree /path/to/main-worktree \
  --checkpoint /path/to/target-checkpoint
```

Run receipt-script unit tests with:

```sh
python3 -m unittest discover -s tools/bench/mtp2485 -p 'test_*.py'
```
