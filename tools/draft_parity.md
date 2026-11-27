# Device draft/plain token parity

`draft_parity.py` runs the same native server binary twice on one local checkpoint: first with its default drafts, then with `--no-drafts`. Every request body is identical between processes. It uses the existing `/v1/chat/completions` and `/v1/completions` routes and shared serving/lane path on Metal, CUDA and other native backends.

The default matrix has two prompts, sampled seeds 0, 1 and 1234 at temperature 1.0 / top_p 0.95, and one greedy cell per prompt. `top_k` and `min_p` are zero. Exact generated token-ID arrays must match; decoded text and token hashes are not the equality oracle. `ignore_eos` requests the complete token budget in every cell. Each cell must have at least one accepted draft in the default run, and zero proposed/accepted draft work in the plain run. A checkpoint with `drafts: none`, proposals but no accepts, or a single inactive cell cannot pass.

This is a correctness check, not a benchmark or a fix for a sampler/model bug. It does not qualify every draft mechanism, precision mode, concurrency level or tool-call parser. A pass means only the recorded matrix passed on that binary, checkpoint and device.

## Run on Metal

Build the native server with Xcode's Metal toolchain, then run on the Mac being tested:

```sh
zig build native -Dcpu=apple_m1
python3 -B tools/draft_parity.py "$MODEL_METAL" --binary zig-out/native/bin/tensorfold-native --output parity-metal --require-mtp
```

For #5223, use the complete local `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` revision `d9d758fb83953437f7263256b0d96157e2a348b8`, including its MTP tensors. `--require-mtp` refuses a `copies only` checkpoint even if copies were accepted; it requires the actual startup `MTP` source label. Acceptance counts still gate every case. This establishes that MTP is available, not how many accepted tokens came from each proposer. Keep the startup logs when reporting a result. The same command can run on an M1 Max, M3 Ultra or M5 Ultra; results on one chip do not qualify another.

## Run on CUDA

Build the server with qualified kernel assets for its GPU. For GB10 sm_121:

```sh
zig build -Dgpu=spark -Dfatbins="$FATBINS"
python3 -B tools/draft_parity.py "$MODEL_CUDA" --binary zig-out/native/bin/tensorfold-native --output parity-cuda --require-mtp
```

`MODEL_CUDA` must be a locally supported CUDA checkpoint with its MTP weights. Do not treat a Metal MLX checkpoint or a CUDA copy missing its MTP tensors as equivalent. The two arms within a run use one binary and one checkpoint; this tool does not compare Metal tokens to CUDA tokens. For families whose default proposer is not MTP, omit only `--require-mtp`; accepted drafts are always required.

## Prompts and settings

The only required positional argument is the local checkpoint directory. The output directory must be new; existing evidence is never overwritten. A free loopback port is selected, both servers bind only to loopback, HTTP proxies/redirects are disabled, update checks are disabled, and the tool stops only its own process. The two processes share the same environment and working directory, run sequentially with parallel=1 and prompt cache budget zero. No model is downloaded.

Provide saved prompts as a JSON object. Nonempty text becomes one user chat message through the checkpoint template; nonempty token-ID arrays go directly to the raw completion route:

```json
{"summary": "Fasse den folgenden Text zusammen: ...", "rendered-chat": [1, 42, 73]}
```

For a chat reproduction, use its already rendered prompt IDs so the two arms see the same model input. The token-ID route skips the template; this tool does not validate tool-call syntax. Pick a context that fits the prompt plus reply. For a longer reproduction:

```sh
python3 -B tools/draft_parity.py "$MODEL" --binary "$BIN" --prompts prompts.json --context 32768 --tokens 256 --seeds 17,23,1234 --require-mtp --output parity-long
```

`--kv-cache TYPE` and `--drafter LOCAL_DIRECTORY` pass the same settings to both processes when needed. Defaults otherwise remain the native server's. There is no switch to waive equality or accepted-draft evidence. `--startup-timeout` and `--timeout` bound startup and each request. Startup labels, health backend/device/version, binary SHA-256, config/index/tokenizer hashes and root weight metadata are recorded. The weight guard is metadata, not a full checkpoint-content hash; freeze the checkpoint and runtime environment for the run.

The registered tool entry is:

```sh
zig build device-draft-parity -- "$MODEL" --binary "$BIN" --output parity-run
zig build test-draft-parity-host
```

The device step is opt-in and is not included in CPU host-test aggregates. It does not build a server implicitly. The host step uses synthetic responses and mocked owned-process transport; it requires no device, checkpoint, listener or network access.

## Results

`report.json` is authoritative together with the process exit status. Exit 0 means every cell is complete, token-identical and has accepted-draft evidence. Exit 1 means a token divergence was observed. Exit 2 means setup, protocol, incomplete generation or missing draft evidence; exit 130 means interruption. If divergence and setup problems coexist, divergence remains visible and the result still fails.

The tool prints draft/accept counts per case and per mode, startup draft labels, and the first differing ID/index. The evidence directory retains both commands, startup logs, all request/response JSON and observed token arrays. Missing counters, invalid IDs, wrong effective draft/sampling modes, wrong health identity, binary/config changes, partial runs and transport errors cannot become a pass. Neither matching text nor merely proposing drafts is sufficient. For `drafts: none` or zero accepted counts, supply a suitable checkpoint/prompt/token budget and run into a new evidence directory; do not label that attempt a parity pass.

Acceptance counters describe verified proposal rows before final output-budget clipping. They need not be bounded by the returned token count, but must be valid integers with accepted no greater than proposed.
