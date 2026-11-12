# Paged FP8: upstream integration resolutions

The cache-only implementation is `684f0cc8881760533bc6591a137c15013bc2b92d` on main
`6c5cef8941b27d86b06a986834e13deb9b515e1e`. The review branch adds this receipt without changing its native code.

## Cumulative latent-split proposals

The public ancestry chain is #2415 (`6398a9627eebf8ed33626aab8ee4ad0b75986582`),
#2935 (`f774e49fa4e47b8c73d0d2cec968aa08c0845561`),
#2937 (`34ae6cb4f2d33c46cc33bc24feca511795c8c205`) and
#2938 (`3e12528aa32db8aa8b07a56885b43de02d0f53a8`).

Their six shared page-format/table/snapshot paths are resolved in
[compatibility merge 72336ffc](https://github.com/gilp456/TensorFold/commit/72336ffc1c040698a44969aeef4657424ea3618f),
whose parents are the cache implementation and the last cumulative split head above.
All four heads merge cleanly with that resolution. Existing contributor commits and authorship are retained.
The cache-only review branch keeps the separate split feature out of its own base and scope.

The resolution retains format-aware FP8 pages with splitting disabled, plus BF16 owner classes, distinct
latent/pooled-key tables, logical token positions and owner-aware snapshots.
The unsupported combined FP8-plus-split mode returns `KvSplitFp8Unsupported` before engine/device construction,
including direct layout/page/state entry points. This admits no unqualified combined format.
The three modified CUDA headers are byte-identical to the cache implementation; this merge changes no kernel.

## Placement proposal

The apparent overlap with #2919 (`68c6d2deeaf24348995a4c5c98a3d653ca0c0627`) is inherited.
Merging that head with either clean main or the cache implementation yields the same seven conflicting files
and ten conflict hunks. The native `finishRank` body is unchanged; its conflict is the existing
`installGroup(&own.group)` versus `installGroup(&own.group, o.launch_from)` call.
The cache's independent paging-test correction introduces no conflict.
The compatibility merge likewise adds no placement conflict paths relative to its split parent.

## Bounded verification

At compatibility merge `72336ffc`, 108 existing/new host tests passed: startup 12, FP8 KV 26,
snapshots 29 and existing split cases 41. Both startup and direct-entry rejection regressions were observed
failing before the unsupported-combination check, then passing afterward.
Native-open and the FP8-reference host compile checks passed; all 18 selected build steps succeeded.
The nine resolved/adapted test/source files match the tested bytes; formatting and diff checks pass.

These public targets reproduce the 67 named-target cases and both compile checks:

```sh
zig build test-glm-startup-host test-glm-fp8-kv-host test-glm-snapshots-host \
  compile-glm-native-open compile-glm-fp8-gate-host -j2 --summary all
```

The additional 41 split cases were selected with a temporary build filter during this bounded verification.
The public, broader `zig build test-cuda-host` includes those cases through the unfiltered GLM host module;
that broader command was not rerun for this compatibility merge.

This is host integration evidence. It adds no device/model/performance qualification and no change to the
separately pinned, qualified release 1.2.7 source or its recorded results. Remaining release gates stay explicit
in the cache proposal. Maintainer/contributor coordination is recorded on [issue3062](https://github.com/ashhart/TensorFold/issues/3062).
