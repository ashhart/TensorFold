# Publication-only checks, 23 October 2026

These are archive integrity checks, not newly run inference or GPU qualification.

- Source base and loader dependency are separated from local experimental increments.
- The six patches applied sequentially to the pinned main base without rejects.
- After applying all patches, 3383 selected native source/build files compared
  byte-for-byte equal to the corresponding frozen v9 source snapshot, with zero
  missing files and zero differences. Selected types: Zig, CUDA, headers, Metal,
  build manifests and the Zig-version pin.
- `git diff --check -- zig build.zig build.zig.zon .zig-version` on the replay
  tree exited 0. This is not a full native build or a lean-check qualification.
- The served-comparison script's CPU self-test passed, including deliberately
  mismatched settings, tokens and missing evidence.
- Recomputing the v5/v9-off/v9-on/reverse-v5 comparison from the sanitized
  records reported all comparisons qualified, with zero error runs. Here
  "qualified" means its listed recorded-evidence comparison checks only,
  not the upstream kernel or model gates.
- The public tree excludes weights, model shards, profiler binaries, credentials,
  deployment controllers and real user prompt/reply bodies. Supplied prompts
  are the public synthetic benchmark fixtures.
- Recorded runtime hashes are preserved rather than presented as independently
  recomputed hashes of unbundled generated output token arrays.
- Code and receipts remain experimental. No new GPU run, service stop, restart,
  model load or device configuration change was performed for publication.

The final `SHA256SUMS` covers every bundled file other than itself. Private source
archive/image identifiers in provenance identify historical measured builds;
those archives/images are not included. Sanitized labels are not registry URLs.
