# Living Weights keep-rate experiment (#1587)

Run this on the issue's untaught Nemotron 3.5 Lightning MLX 4-bit checkpoint with the base binary and the candidate binary.
Use the same backend, device, runtime libraries and build settings. Run it with exclusive use of the GPU.
The runner starts one server process at a time.

```sh
python3 tools/zig/keep_rate_ab.py \
  --base BASE_BINARY --candidate CANDIDATE_BINARY \
  --checkpoint CHECKPOINT --checkpoint-revision REVISION \
  --backend mlx --output NEW_RECEIPT_DIRECTORY
```

For a GB10 use `--backend cuda` and matching CUDA binaries/checkpoint. `--server-args` accepts a JSON argument
array, without a shell. Its default is `['--context','4096','--parallel','2','--temperature','0']` (use JSON double
quotes when overriding). Host, port, authentication, slide, seed-policy and thinking flags are managed by the
runner. It rejects overrides of those flags and quiet logging. It clears `TF_SLIDE_*` environment overrides for
both arms. Other runtime environment settings are inherited and must match; save them with the device receipt.
`--port`, `--startup-timeout` and `--request-timeout` can be supplied. Binaries must directly accept `serve MODEL`.

The fixed lesson is `{"text":"My favourite colour is teal."}` with no supplied wordings or explicit request seed.
Each binary runs ten fresh trials with `--prompt-seed` and ten without it, under default thinking and
`--no-thinking`: 40 lessons per arm, 80 total. The two arms alternate order between successive trials. Each trial
starts from its own independent checkpoint copy and empty graph. Clone/reflink copies may share physical pages
but never file inodes. The runner checks the copied file set and sizes and hashes the original before and after.
The input checkpoint must already be untaught; an empty metadata graph alone cannot prove that its weights are.

Each arm/mode/seed-policy cell first records a separate fresh `--slide`-off token baseline. Each lesson trial
records those same unrelated requests before and after teaching with `--slide` on, then restarts that copy with
`--slide` off and records them again. The controls ask three fixed unrelated questions greedily, with drafts off,
fixed request seeds and full generated token arrays, including reasoning tokens. Their requests use the server's
thinking default. Token IDs must be valid u32 values and match the server's hash when provided. Any mismatch is
reported. Sampled, drafted, concurrent and resumed serving comparisons remain separate device gates.

Receipts include binary and runner hashes, source checkpoint hashes/revision, commands, process IDs/exits, health,
complete server logs, raw lesson SSE, parsed events, graph snapshots, every parsed round and cumulative step count,
per-fact plain/thinking/probe counts, redraws and actual logged sampled-request seeds. Seed records name the
request and log line. A round also lists seeds logged since the preceding round; this includes writer and check
calls and does not pretend to identify a seed for a greedy request. Partial HTTP bodies and available round logs
are retained when a request fails. A stopped experiment's incremental results remain on disk.

Older base binaries lack the new counts. These are `null`, with `rounds_missing_counts` reported, never inferred
from the verdict or filled with zero. Missing candidate counts or absent round/seed logs fail the receipt.
Per-round `recalled` and `held_token_equal` preserve the server's own verdicts. Final keeps require a completed
stream, successful save, recalled learned events and a learned graph fact. This runner does not independently
reimplement the server's acceptance checks or establish that content seeding improves the real model.

`summary.json` includes every scheduled trial in each keep-rate denominator. Refusals remain in `results.json`
and their SSE files. Transport/protocol failures are errors, not invented refusals. A known lesson verdict remains
in the rate if a later measurement fails; the error count can overlap that known keep/refusal. Missing trials and
unresolved lessons are explicit. Exit zero means the protocol completed and recorded token controls agree, even
if every lesson was refused. It is not a release qualification or an improvement claim.

Checkpoint-copy hashes after each case list changed files. Copies are removed after recording unless
`--retain-copies` is supplied; use it when inspecting fitted neurons or failed restoration. Full weight hashing
and restart checks add experiment time. `lesson_seconds` measures only the learn HTTP request; `case_seconds`
also includes copy, startup, controls, restarts, hashing and cleanup. Keep logs and manifests for all attempts.

The host test launches only synthetic child HTTP servers with tiny fictional checkpoint files:

```sh
python3 -m unittest discover -s tools/zig -p test_keep_rate_ab.py -v
```

It executes the full 80-trial matrix and checks refusals, flags, raw receipts, exact tokens, restart controls,
original-file safety, unknown old counts, missing candidate diagnostics, truncated streams, HTTP errors and
occupied ports. It neither loads model weights nor opens a GPU driver. Real keep rates remain unmeasured here.
