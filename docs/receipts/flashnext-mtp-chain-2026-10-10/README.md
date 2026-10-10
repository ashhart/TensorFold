# Flash Next batch MTP chaining receipt

The shared Metal adapter can submit a stream's kept-window absorb and all chained MTP drafts in one command buffer.
`FZ_BATCH_MTP_CHAIN=1` enables the experiment; it defaults to off. Streams still draft serially.
Each encoded step owns its CPU-written position, row and selector metadata. GPU score/key scratch remains shared.
The new buffers add 17,472 bytes of payload per session; admission still checks the actual process footprint.

## Qualified scope

TensorFold 1.0.5, upstream `bec00ae6961a43a9ab73c4fa9338c167b3f82851`;
inference source `6dbf17059a846f2c9c40a92c79b5558d41f2960f`.
The [manifest](manifest.json) pins the source tree, compiler archive and both executed binaries.
Rebuilt binaries match the hashes of the executed artifacts.

One 80-core M3 Ultra, 256 GiB RAM, macOS 26.7.1, system Metal, Zig 0.17.0 targeting `apple_m1`.
Checkpoint `TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP`, download metadata revision
`45032a193a5d4126ba8534822cbaf8fb30b3e368`.
Public fixtures have 23, 66, 2,764 and 11,237 prompt tokens, with thinking off and on.
The direct gate also uses the first 2,050 tokens of the longest fixture to cross into sparse selection during drafting.

## Correctness

[Checkpoint gate](checkpoint-gate.log):

- 180 paired comparisons at absorb widths 1/4/16, previous speculative tails 0/3, and draft depths 0/1/2/4/8/15.
  Draft tokens, final hidden-state bytes, head positions and pooled-block counts match serial drafting exactly.
  Wider absorbs use controlled repeated row inputs to exercise the head independently of the scheduler.
- Four 64-token replies match fresh legacy plain decoding in plain, serial-MTP and chained-MTP modes.
- All three modes pass cancellation, slot reuse, prompt admission, multi-chunk marks and prefix restoration with an active peer.
- [HTTP receipt](http.json): 96 responses across six blocks, four rounds each. Bodies, token fingerprints and usage match.
  Every four-slot block observes four active sessions. Prefix caching is disabled.

The executed HTTP helper is pinned to the inference-source commit.
The final helper additionally invalidates stale success before reading fixtures and rejects unobserved shared sessions.
Both failure cases were checked locally; the recorded run independently satisfies the strengthened parallelism gate.

## Timings

The checkpoint gate records one observation, including prefill and decode:

| Phase | Serial MTP | Chained MTP |
| --- | ---: | ---: |
| Draft calls | 84 | 84 |
| GPU waits during drafting | 256 | 84 |
| Draft time | 0.312223 s | 0.261944 s |
| Total group time | 12.473931 s | 12.375544 s |

Draft time fell 16.1%. This single observation does not establish a general speedup.

HTTP uses identical public requests, greedy decoding, a 16,384-token context and 128-token reply limit.
The one-slot control runs off/on. Four-slot blocks run off/on/on/off.
Each block has one excluded warmup and three measured rounds; every group emits 459 tokens.

| Parallel slots | Serial warm median | Chain warm median | Group throughput change |
| --- | ---: | ---: | ---: |
| 1, legacy control | 15.191969 s | 15.226884 s | -0.23% |
| 4, shared adapter | 14.599447 s | 14.496549 s | +0.71% |

These are end-to-end group timings, including prompt processing. They are not an isolated decode benchmark.
The effect is small; the option stays experimental. The one-slot driver does not consume the switch.

## Memory and cleanup

A [footprint sample](memory.json) from the final four-slot serial block reports a process-lifetime physical peak
of 186,421,791,496 bytes. This is not a whole-run peak across all server restarts.
System swap usage remained zero at the probes. One model process ran at a time under the existing admission lock.
The original service was restored and verified healthy; native test processes and the lock were removed.

## Host validation and limits

[Host check](host.log): 137/137 build steps succeed; 351 tests pass, two are skipped and none fail.
Other test steps are cached. Whole-tree `zig fmt --check build.zig zig` passes with Zig 0.17.0.
[Golden check](golden.log): 309 equal, 10 documented differences, zero unexpected differences.
Both native builds pass. The lean checker finds 542 inherited problems on upstream and candidate, with none added.
Astra reviewed the chain, the metadata lifetime, state accounting, direct gates and benchmark failure handling.

This receipt qualifies these fixtures on one M3 Ultra and this checkpoint.
CUDA, other Apple chips, other RAM classes/checkpoints, more than four sessions, 32K+ contexts,
separate cold-prefill sweeps, a previous-release comparison and a standard-server comparison were not run.
The patch is a shared-adapter scheduling change using the existing Metal operators and precision.

## Reproduce

Run on an idle GPU with no other model loaded and use the platform's admission lock and service restoration procedure.

```sh
zig build native tf-flashnext-batch -Dcpu=apple_m1 -Doptimize=ReleaseFast
FZ_N=64 zig-out/bin/tf-flashnext-batch "$MODEL" \
  docs/receipts/flashnext-mtp-chain-2026-10-10/prompts/{short,thinking,sparse,sparse-thinking}.json
python3 tools/zig/flashnext_mtp_chain.py \
  zig-out/native/bin/tensorfold-native "$MODEL" \
  docs/receipts/flashnext-mtp-chain-2026-10-10/prompts output --tokens 128 --rounds 4
zig build test test-golden -Dcpu=apple_m1
zig fmt --check build.zig zig
```
