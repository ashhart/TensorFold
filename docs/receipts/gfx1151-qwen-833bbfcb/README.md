# gfx1151 Qwen qualification at TensorFold 1.0.12

Single-GPU model and HTTP equality receipt for [#426](https://github.com/ashhart/TensorFold/issues/426).
Measured on 2026-10-17, on a BOSGAME M5 with an AMD Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`).
The tested source is **`833bbfcba9109bb38c7af7303f80da0ca12b213c`**, tagged `v1.0.12`, with a clean tree.
This receipt does not qualify later `main` commits, other model families, or multi-GPU configurations.
No inference implementation changes are included.

## Environment and checkpoints

- Zig 0.17.0; Ubuntu HIP 7.1.52801-9999; Ubuntu clang 21.1.8.
- The HIP build used `ReleaseFast`, explicit `gfx1151`, and the LLVM 21 device bitcode directory.
- Binary SHA-256 values and the complete build command are in [provenance.json](provenance.json).
- All 48 local checkpoint files checked against public Hugging Face metadata matched their pinned revisions.
  [checkpoints.json](checkpoints.json) retains each Git blob or LFS SHA-256 digest, size and match result.

| Checkpoint | Public repository | Revision |
| --- | --- | --- |
| Qwen3.5-9B, MLX 4-bit | `lmstudio-community/Qwen3.5-9B-MLX-4bit` | `b455506b0f574c74616dbcd56879bde38fafcff3` |
| Qwen3.6-35B-A3B, MLX 4-bit with MTP | `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP` | `f84b054c5677b6e59bdb969ae61915b0983d68bb` |
| Qwen3.8-27B, MLX 4-bit | `TensorFold/Qwen3.8-27B-MLX-4bit` | `70ae7fac63274ff2eac54152031433374cb80f2f` |

## Results

| Checkpoint | Model invariants passed | Failed | Skipped | HTTP comparisons exercised and passed | HTTP drafting comparisons not exercised |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B | 42 | 0 | 5 | 13 | 2 |
| Qwen3.6-35B-A3B + MTP | 46 | 0 | 1 | 15 | 0 |
| Qwen3.8-27B | 42 | 0 | 5 | 13 | 2 |
| Total | **130** | **0** | **11** | **41** | **4** |

The unmodified `tf-qwen35-test check` gate exercised row/residual equality, concurrency, graph/eager equality,
resume and radix behavior. Its exact cases and results are retained in `*-model-check.log`.

The unmodified `serve_check.py` exercised solo/concurrent equality, long-prompt interleaving, repeatability,
and its resumed/fresh comparison. It ran under [record-served.py](record-served.py), which wraps `urllib.request.urlopen`
to append each request and response to the `*-serve-check.jsonl` file; for the standard `serve_check.py` run it does not
change request bodies or replies. It also skips the optional extended suites when a `skip-extended` marker exists (below). Greedy and sampled requests use the same public prompts across models.
The HTTP logs retain all fifteen printed `ok` lines per checkpoint, but **45 printed successes do not mean
45 exercised comparisons**: the 9B and 27B have no draft head, so their two drafting comparisons are vacuous.
Those four comparisons are classified as not exercised in the table and [summary.json](summary.json).
The other thirteen HTTP comparisons still execute on those checkpoints.

The 35B requests did draft. In `35b-serve-check.jsonl`, all **36** drafted replies accepted drafts,
**974 of 1593** drafted tokens in all, from each reply's `speculative` telemetry rather than its `draft` flag.
The server log's accepted/drafted counts **30/68, 31/33, 26/46 and 34/51** are the four drafted == serial comparisons.
Both greedy and sampled drafted replies matched plain replies. Drafted == plain covers this one checkpoint only:
the 9B and 27B have no draft head.

The three `*-serve-check.jsonl` files retain all **183 HTTP 200 responses**, their request bodies, complete replies,
cache/speculation telemetry, and native full token SHA-256 values. Independently grouping identical request bodies
while ignoring only `draft` gives **33 groups with zero token-hash mismatches**.
[http-token-hash-audit.json](http-token-hash-audit.json) records that audit separately from the scripts' printed results.

### Skips and remaining coverage

- Eight model drafting controls did not execute: four each on the 9B and 27B, which have no draft head.
- All three model gates skipped `radix memory: too few ids` at the default 1536 synthetic IDs; the control needs 1548.
  It was run afterwards with 2048 IDs and passed on all three checkpoints (see Follow-up below).
- The four unexercised HTTP drafting comparisons above are separate from those eleven model skips.
- Extended `serve_rewind.py` ran on the 9B for 9 exchanges and was stopped (`rc=-15`); it is incomplete and unqualified.
  The final requests in `9b-server-8092.log`, ending with one marked `reason=client-left`, are from that stopped run.
  For every later run, `record-served.py` found its `skip-extended` marker and exited 77 before starting `serve_rewind.py`
  or `structured_served.py`, so neither upstream script ran on the 35B or 27B, nor `structured_served.py` on the 9B.
  Neither suite is qualified, and the full structured-output matrix is not covered.
- Refusals: 0 of 183 HTTP requests were rejected (all returned 200). Content refusal behavior was not tested,
  and no deliberate refusal case (for example a prompt past the context window) was sent.
- No independent cross-engine model-accuracy reference, long-context performance comparison,
  multi-GPU run, or other model family is qualified by this receipt. Existing model-free runtime tests were not repeated.
- This is an equality/correctness receipt. Raw timing fields are retained, but no speedup or performance-regression claim is made.

## Reproduction

Use the tested commit and the checkpoint revisions above. Run from that pinned checkout.
`ZIG_DIR` points to the Zig 0.17.0 installation; `MODEL` points to one checkpoint directory.
The same build produced both binaries for all three models:

```sh
"$ZIG_DIR/zig" build native-hip tf-qwen35-test \
  -Dhip-arch=gfx1151 -Dhipcc=/usr/bin/hipcc -Dhip-include=/usr/include \
  -Dhip-device-lib=/usr/lib/llvm-21/lib/clang/21/amdgcn/bitcode \
  -Doptimize=ReleaseFast -Dversion=1.0.12 -j4
./zig-out/bin/tf-qwen35-test check "$MODEL"
```

Each model then ran in its own single server process with these flags:

```sh
./zig-out/native-hip/bin/tensorfold-native serve "$MODEL" \
  --backend rocm --host 127.0.0.1 --port 8092 --name "$NAME" \
  --context 16384 --parallel 4 --prompt-cache-gib 1 --no-thinking
python3 "$RECEIPT_DIR/record-served.py" zig/tests/qwen35/serve_check.py \
  "$RECEIPT_DIR/$TAG-serve-check.jsonl" http://127.0.0.1:8092 "$NAME" 64
```

`serve_check.py` can also be run directly with the same last three arguments; the recorder only keeps the exchanges.

`NAME` was `amd-9b`, `amd-35b` or `amd-27b`, and `TAG` was `9b`, `35b` or `27b`. Requests use `max_tokens=64`, temperature 0 or 0.8,
seed 7, and `enable_thinking=false`; exact bodies are in the JSONL files.
The initial model gate uses its upstream default synthetic IDs and flags; it does not include the optional truth/accuracy phase.
The successful build was quiet; native execution and the binary hashes identify the executables actually exercised.

## Follow-up: radix memory and cancellation

Run the same day with the same binaries, checkpoints and serve flags.

| Checkpoint | Invariants with 2048 IDs passed | Failed | Skipped | Radix checks passed | `serve_cancel.py` |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-9B | 43 | 0 | 4 | 8 of 8 | 3 ok, failed 0 |
| Qwen3.6-35B-A3B + MTP | 47 | 0 | 0 | 8 of 8 | 3 ok, failed 0 |
| Qwen3.8-27B | 43 | 0 | 4 | 8 of 8 | 3 ok, failed 0 |

The remaining skips are the drafting controls of the two checkpoints without a draft head.
[ids2048.npy](ids2048.npy) holds `1000 + (i * 7919) % 50000` for `i < 2048`, the upstream synthetic formula made longer.
After each cancelled long prompt, the next short request took as long as one alone (9B 0.84 s and 0.84 s,
35B 0.21 s and 0.20 s, 27B 1.79 s and 1.88 s).

```sh
./zig-out/bin/tf-qwen35-test check "$MODEL" --ids ids2048.npy --only invariants
python3 zig/tests/qwen35/serve_cancel.py http://127.0.0.1:8092 "$NAME"
```

## Published evidence

- `*-model-check.log`: upstream model-gate output, including every skip.
- `*-serve-check.log` and `*-serve-check.jsonl`: upstream HTTP output and captured exchanges.
- `*-server-8092.log`: server startup, cache and accepted-draft telemetry.
- `record-served.py`: the recorder the HTTP checks ran under.
- `*-radix-memory-check.log`, `*-serve-cancel.log`, `*-server-8092-cancel.log`, `ids2048.npy`: the follow-up runs.
- `summary.json`, `provenance.json`, `checkpoints.json`, `http-token-hash-audit.json`: structured results and provenance.
- `publication.json`: source/published file hashes and private-path replacement counts.
- `SHA256SUMS`: digests of every published receipt file except the manifest itself.

This is a separate sanitized copy. Private checkout/model paths and the machine's local hostname are replaced
with placeholders; trailing whitespace in text logs is normalized. Request/response values, tokens, native token hashes
and telemetry are preserved.
Machine-local coordination/service-control files and incomplete extended-run exchanges are omitted.
The original local receipt is unchanged. No private TensorFold Notes code, model weights or credentials are included.
