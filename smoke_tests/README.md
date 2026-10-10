# Security review — `cuda-checkpoint-cli` vs `main` (Reviewer3)

Status: fixes for F1–F4 applied by the team-lead; the two traversal tests now assert the
refusal (`snippetPathOk` in `pull_file.zig`, 40-hex sha gates in `pull.zig` and both `hub.zig`
sinks), and both keep positive controls. Commands updated below.

Scope: the single commit `d5afccf` — `zig/src/cli/cli.zig`, `models.zig`, `info.zig`, `pull.zig`,
`pull_file.zig`, `pull_parts.zig`, `hub.zig`, `cuda_main.zig` dispatch, `zig/build/cuda.zig`, `build.zig`.

## Trust boundary map

```
 argv (user) ──► cli.main / cuda_main dispatch ──► pull/info/models
                          │
 env: HF_ENDPOINT, HF_HUB_CACHE, HF_HOME, HOME (user-controlled) ──► hub.cacheDir, endpoint base URL
                          │
     ════════ NETWORK BOUNDARY (TLS via std.http.Client) ════════
                          │
 hub API JSON: revision "sha", tree entries {path,size,lfs.oid,oid}, config.json bytes, file bytes
                          │
 ──► sinks: std.fs.path.join / createDirPath / writeFile / symLink / deleteFile / renameAbsolute
            under $HF_HUB_CACHE (or ~/.cache/huggingface/hub)
```

Untrusted actors: (a) whoever serves `HF_ENDPOINT` responses — the normal hub, a mirror, or
anything the env var names; (b) repo publishers via tree/config JSON; (c) anything that can
write the user's cache directory (local attacker, another tool). `repo` is well validated
(`hub.isRepoIdLike`); network JSON strings are **not**.

## Findings

### F1 — MEDIUM — Hub-controlled file path traverses out of the cache
`pull.zig listTree()` copies `path` from the hub's tree JSON verbatim; it reaches
`pull_file.linkIntoSnapshot(a, io, snapshot_dir, path, blob)` which does
`std.fs.path.join(&.{ snapshot_dir, path })` (pull_file.zig:~118) — no `..`/absolute check — and
then `symLink` plus a `deleteFile` on that path. A tree entry `"path": "../../outside/evil"`
plants a symlink outside the cache (and the `deleteFile` can unlink an arbitrary user file first).
Preconditions: malicious/compromised endpoint or a repo whose tree carries `..` (git itself
forbids `..` components, so a real HF tree cannot hold them — this is why not High). Proven:
`r3_linkIntoSnapshot_traversal.zig` (passes).

### F2 — MEDIUM — `sha` from the revision JSON and `refs/main` are joined into paths unvalidated
`pull.zig resolveRevision()` returns the JSON `sha` string raw (only non-empty is checked); it is
used as `snapshots/<sha>` (pull.zig:~120) and written into `refs/main` (pull.zig fetchBig
success path). `hub.cachedSnapshot` then joins the `refs/main` content (trimmed, ≤256 bytes)
into a path: `join(root, "snapshots", rev)` — a revision like `../../escaped-dir` resolves the
"serving" snapshot outside `models--Org--Name`, so subsequent reads (config.json, weights) and
writes come from an attacker-chosen directory. Preconditions: an endpoint under attacker
control, or any writer of the cache dir. Proven on the real `hub.cachedSnapshot`:
`r3_cachedSnapshot_ref_traversal.zig` (passes). The `snapshots/<sha>` join in pull.zig is the
same primitive (same `std.fs.path.join`, no validation between the JSON and the join).

### F3 — LOW — `REVISION` in `REPO[@REVISION]` is not validated before URL interpolation
`repo` is run through `isRepoIdLike`, but `revision` goes raw into
`{endpoint}/api/models/{repo}/revision/{revision}` (pull.zig resolveRevision) and
`.../resolve/{sha}/{path}`. A revision containing `/`, `?`, or `..` reshapes the request path.
Impact is limited to requests against the configured endpoint (no SSRF to a new host by itself);
still, one `isRepoIdLike`-style check would close it. No smoke test (sink is a URL string).

### F4 — LOW — Unbounded allocation from a network-controlled size on resume
`pull_file.download()`: `resume_from` is capped to `e.size` (from tree JSON, can be huge), then
`a.alloc(u8, @intCast(resume_from))` allocates the whole prefix in memory to hash it. A
multi-GiB `e.size` plus a pre-planted partial file (local writer) forces an OOM on 64-bit.
`@intCast` also panics on 32-bit if `resume_from > maxInt(usize)`. Streaming the prefix through
the hash in chunks removes it.

### F5 — INFO — Files with no digest get no integrity check
A tree entry with neither `lfs.oid` nor `oid` yields `sha256 == null && git_sha1 == null`;
`download()` then writes it with no verification and `blobName` falls back to the path. Same as
upstream huggingface_hub behavior; noted for completeness.

### Positives (checked, not findings)
- `hub.isRepoIdLike` rejects traversal shapes in `repo` (proven in `r3_negative_controls.zig`).
- `blobName` hex-ifies digests and replaces `/` in path fallbacks — blob names cannot traverse
  (proven in `r3_negative_controls.zig`).
- sha256 is computed over the whole file (prefix + network) before the blob is renamed into
  place; `pull_parts` verifies the 206 `Content-Range` start byte matches the request.
- Range-unsupported fallback re-downloads whole files and re-hashes; mismatched bytes are
  discarded, never linked.
- TLS through `std.http.Client` (certificate verification on by default); `identity` encoding
  pinned so sizes/digests match the wire bytes.
- `config.json` is fetched and family-checked before any weight byte moves; unknown model_types
  are refused.
- `readSmall` caps config reads at 16 MiB; `cli.zig` arg counts are exact per subcommand.

## Severity framework applied
- Critical: unauthenticated RCE / arbitrary code from normal use — none found.
- High: needs no unusual preconditions — none found (F1/F2 both require a hostile
  `HF_ENDPOINT`, a compromised hub, or a local cache writer; git also forbids `..` path
  components in real trees).
- Medium: F1, F2 — real traversal primitives on attacker-supplied strings, mitigated
  preconditions as above.
- Low: F3, F4. Info: F5.

## How to run the smoke tests

```sh
zig test --dep pull_file -Mroot=smoke_tests/r3_linkIntoSnapshot_traversal.zig \
         -Mpull_file=zig/src/cli/pull_file.zig
zig test --dep hub --dep native_engines \
         -Mroot=smoke_tests/r3_cachedSnapshot_ref_traversal.zig \
         -Mhub=zig/src/cli/hub.zig -Mnative_engines=smoke_tests/r3_native_engines_stub.zig
zig test --dep pull_file --dep hub --dep native_engines \
         -Mroot=smoke_tests/r3_negative_controls.zig \
         -Mpull_file=zig/src/cli/pull_file.zig -Mhub=zig/src/cli/hub.zig \
         -Mnative_engines=smoke_tests/r3_native_engines_stub.zig
```

All three pass (zig 0.17.0). The two traversal tests now prove the sink is safe: F1's asserts
`error.SnapshotPath` on `..`/absolute/dot-component paths and no symlink planted outside, plus a
positive control that a nested dotted name still links; F2's asserts `cachedSnapshot` returns null
for traversal/short/non-sha refs and still resolves a real 40-hex ref. The negative controls pass
(needs all three module deps on the command line now). The stub (`r3_native_engines_stub.zig`)
only replaces the engine registry import of `hub.zig`.

## Fixes applied
- F1: `pull_file.snapshotPathOk` — `linkIntoSnapshot` refuses empty, absolute and
  `.`/`..`-component paths with `error.SnapshotPath` before any `deleteFile`/`symLink`.
- F2: `pull.zig hexSha` — the revision JSON's `sha` must be 40 hex; `cli/hub.zig` (`cachedSnapshot`)
  and `server/hub.zig` (`resolve`) apply the same gate to `refs/main` content, failing closed to the
  newest-with-config fallback / a refusal.
- F3: `pull.zig revisionOk` — the `REVISION` of `REPO[@REVISION]` must be alphanumeric with `-_.`,
  no `..`, checked before the URL is built (own error message, no request leaves).
- F4: `pull_file.download` reads and hashes the on-disk resume prefix in 64 KiB chunks; no whole-file
  allocation from a hub-named size, no `@intCast` on a network value.
- F5 (INFO, unchanged): entries with no digest keep huggingface_hub's behavior; no diff.

## Suggested fixes (one pattern each)
- F1: reject `path` entries containing `..` components, empty, or absolute before entering
  `Big`/download queues (validate once in `listTree`).
- F2: validate `sha` as `[0-9a-f]{40}` (or at minimum no `/`, no `..`, no leading `.`) in
  `resolveRevision`, and apply the same to `refs/main` content in `cachedSnapshot`.
- F3: validate `revision` like `repo` (allow `@`? strip after first check) before URL building.
- F4: hash the prefix in 4 MiB chunks instead of one allocation.
