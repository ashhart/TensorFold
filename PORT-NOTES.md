# Port notes: E1 + E2 from `tp4-mesh-roce` (f350cd3e + 106 recipe patches) onto upstream `main` 8094ec96

Branch `roce-any-world` = `origin/main` (8094ec96) + two commits:

| Commit | Ported from | Subject |
| --- | --- | --- |
| 5bc0ffba | eeca3586 (E1) | cuda roce: the one-shot RoCE all-gather at any world whose ranks all reach each other, with a connect probe |
| 16fa3ac1 | 5690f334 (E2) | glm cuda: the DFlash2 drafter MLP split across every rank (FF / world), the partials summed in rank order |

Patches: `../roce-any-world.patch` (`git format-patch -2 --stdout`) and `../roce-any-world-patches/000{1,2}-*.patch`
(the per-file form the recipe's Dockerfile applies). Both were dry-run on a clean `origin/main` worktree with
`git apply --check` then `git apply`, in filename order, and the resulting tree is identical to the branch tip.

**No Zig toolchain on this machine**: nothing here was compiled or run. See "Not verified" below.

## What `main` already has, and why E1 still stands

`main` did NOT change `zig/src/cuda/roce.zig`, `roce_gather.cu`, `zig/tests/cuda/roce_gather_tests.zig` or
`docs/roce-gather.md` since f350cd3e; `roce.zig` still refuses `world != 2`. But `main` reaches world > 2 small
gathers for GLM by another route (81e25bb5, b99c48a5, 65ec7ead): `fabric/tp`'s `roce_multi` wire (3-8 ranks, its own
TCP bootstrap at `TF_TP_MASTER`, its own kernel image file `TF_TP_KERNEL_IMAGE`, per-pair port picking that works on a
switchless ring) is lent to `tp_group.Group.multi` behind `TF_GLM_ROCE_MULTI=1`, default off (CHANGELOG: +2.57% decode
at TP=3 ring, cold 136k prompt 1.2-1.6% slower, "so NCCL stays the default"). E1 generalises the simpler, zero-config
path (`roce.zig`: same `TF_ROCE*` knobs, no bootstrap, no kernel image file, no extra proxy design) to any world on a
fabric that pairs the same HCA index across ranks (a switch). Upstream may ask why two multirank wires; the answer in
the commit message is: this one needs nothing but the group's NCCL communicator and is the one measured at +10%.
If upstream prefers one wire, E2 is independent of E1 (its extra gather rides whatever `Group.allGather` has).

## Per-file: pin vs `main`, and how each hunk was adapted

| File | `main` since f350cd3e | Recipe (106 patches) touched it? | Port |
| --- | --- | --- | --- |
| `zig/src/cuda/roce.zig` | unchanged | no | E1 applied verbatim, then two edits (below) |
| `zig/kernels/cuda/roce_gather.cu` | unchanged | no | verbatim |
| `zig/tests/cuda/roce_gather_tests.zig` | unchanged | no | verbatim |
| `docs/roce-gather.md` | unchanged | no | verbatim, then the refuse sentence reworded (below) |
| `zig/build/cuda_kernels.zig` | +42 lines elsewhere | no | comment hunk auto-merged |
| `zig/tests/cuda/main.zig` | +50 lines elsewhere | +9 (recipe subcommands) | usage + dispatch hunks auto-merged; recipe lines not carried |
| `zig/src/cuda/tp_group.zig` | +111: `Multi` borrowed wire, `benchmarkShape`, tests | +13: `p2p` peer-store gather (recipe-only) | conflict. Kept `main`'s `Multi`; dropped the recipe's `p2p` field/`openP2p` (not on `main`); E1's `fast` comment, module doc and `openRoce` kept; `openRoce` doc rewritten for fallback semantics and gained an early return when `world > roce.max_world` (so a 9+ rank group keeps NCCL instead of `Gather.init`'s `error.PeerRefused`) |
| `zig/src/families/glm/cuda_ranks_test.zig` | +129 (the fake `Fabric` with `switched`/`reaches` IS on `main`) | +59 | E1's fabric-contract test auto-merged; one comment reworded (refuse -> every rank keeps NCCL) |
| `zig/src/families/glm/cuda_native.zig` | `openRoce` then `cuda_roce_multi.open` | - | NEW touch (not in eeca3586): `openRoce` is skipped when `TF_GLM_ROCE_MULTI=1`, so an explicitly named borrowed wire is not doubled by `roce.zig`'s path beneath it (`allGather` checks `fast` before `multi`) |
| `zig/src/families/glm/dflash.zig` | +`confidence()`, `Req.propose`, `dflash_proposals.active` | +120 (Q32 scratch etc.) | conflict at `confidence()` vs `mlpSplit`: both kept. `rawBytes` keeps `main`'s `2 * nk / 32` (the recipe's `/16` + `TF_GLM_DFLASH2_Q32` comment dropped) |
| `zig/src/families/glm/dflash_load.zig` | header parsing via `core.safetensors` (`addEntries`), a test | +33 (`gwidth` Q32) | conflict x2: kept `main`'s `addEntries`; `carve` keeps `D / 64` (recipe `gwidth` dropped); only `"sums"` -> `dr.world * M * D * 4` |
| `zig/src/families/glm/dflash_block.zig` | unchanged | +16 | E2's two hunks auto-merged (swiglu grid, `df_sumn` + world arg) |
| `zig/kernels/cuda/glm_dflash.cu` | unchanged | +11 | `df_sumn` auto-merged; removed the stale `df_sum2` comment line the original left above it; swiglu comment says `ceil(F / 1024)` |
| `zig/src/core/contract/glm_cuda_records.zig` | +9 elsewhere | no | `df_sum2` -> `df_sumn` record auto-merged (register figures carried over) |
| `zig/glm_dflash_host.zig` | unchanged | no | verbatim |

### The one behavioural deviation from the original E1 (deliberate)

Original: a failed connect probe returns `error.PeerUnreachable` on every rank and logs `REFUSE TF_ROCE ...`; the
start dies. Port: every rank returns `null` from `Gather.init` (NCCL carries every gather) with one `warn` naming the
rank; the error name stays in the set for `probe()`'s internal return. Reasons:

1. On `main` today `TF_ROCE` unset at world 3+ is a no-op. Upstream ships a 3-GB10 switchless ring (#1516, #1524)
   where each pair sits on different ports; `roce.zig` writes HCA h -> peer's HCA h, so that ring's probe fails. With
   the original semantics it would refuse to boot by default after this change. Fallback keeps it on NCCL as before.
2. Every other failure in `roce.zig` (no verbs, no HCA, RTR/RTS refused, a rank with `TF_ROCE=0`) already falls back
   "every rank or none"; `fabric/tp`'s wire does the same (`TF_TP_ROCE_FALLBACK=error` to refuse). Same shape.

The downstream recipe's GO-C gate ("every rank prints REFUSE and exits non-zero") becomes "every rank prints the
warning and runs on NCCL"; the launcher can grep for `RoCE all-gather: {d} ranks` to know the path is up. If you want
the refuse back, it is the `if (std.mem.indexOfScalar(u32, verdicts, 0)) |r|` block in `Gather.init`.

## Dependencies on the 106 recipe patches

- E1: **none**. Its files were untouched by the recipe except `tp_group.zig` (recipe `p2p`, dropped) and
  `tests/cuda/main.zig` / `cuda_ranks_test.zig` (recipe additions, not carried). The `Fabric` fake E1's contract test
  needs is on `main`. 0106's `warmPeers` is not referenced.
- E2: **none**. Every hunk landed on `main`'s own lines; the recipe-only parts in its context (`gwidth` / Q32 scratch,
  `2 * nk / 16`) were dropped. E2 does not need E1 (the gather goes through `Group.allGather`).

## Dropped

- Recipe `p2p_gather` plumbing in `tp_group.zig` (not upstream).
- `REFUSE TF_ROCE` hard error (see above).
- Nothing else; every test of the original two commits is present.

## Tests kept (adapted to `main`'s layout, none executed here)

| Test | Where | Runs with |
| --- | --- | --- |
| cap over 2 and 4 ranks; two-rank layout == old formula byte for byte; four-rank slots/flags non-overlap, sender's rank order, both sides agree on peer numbers; `@sizeOf(Blob) % 8 == 0`; fabric contract (mesh/ring/cable) | `zig/src/cuda/roce.zig` | `zig build test` (cuda module) |
| same fabric contract on the fake `Fabric` | `zig/src/families/glm/cuda_ranks_test.zig` | `zig build test` (glm) |
| `mlpSplit` at every world 1-16; raw scratch follows it; every `df_*` symbol has a record (`df_sumn`) | `zig/glm_dflash_host.zig` | `zig build test-glm-dflash-host` (or whatever step `main` names; the file is unchanged on `main`) |
| `tp_group` borrowed-wire test (upstream's) still compiles | `zig/src/cuda/tp_group.zig` | `zig build test` |
| `tf-cuda-test roce-gather <rank> [world]` | `zig/tests/cuda/roce_gather_tests.zig` | real ranks, `NCCL_COMM_ID` |

## Not verified without a toolchain / hardware

- Compilation under Zig 0.17.0 / nvcc 13.3 of every changed file. Idioms used (`for (..) |r| if (..) {..};`,
  `@splat(@splat(0))`, `[n]?*T = @splat(null)`, `std.mem.indexOfScalar`, `DeviceBuffer.upload` on a by-value
  `DeviceBuffer`) all have precedents on `main`; `extern struct Blob` is 144 bytes (8-aligned).
- `main`'s `tp_group` test constructs `Group` with `.lib = undefined` and a fake `ncclAllGather`: E1 adds no field without
  a default, so it still builds.
- `df_sumn`'s register record (carries `df_sum2`'s 16/38): `tf-cuda-test contract-limits` on the first image will say
  whether the contract test accepts it or wants the refreshed figures (then paste its `RECORD df_sumn:` line).
- Behaviour of `roce.zig` at world 3+ on fabrics other than the four-Spark switch (the probe's fallback is designed,
  not run); the 5 s probe ceiling on a real unreachable pair.
- The interplay with `TF_GLM_ROCE_MULTI=1` (now mutually exclusive by the `cuda_native.zig` gate) was not run.
- `minimax_m3/cuda_native.zig` also calls `openRoce`; at world > 2 it now gets this path too (unchanged contract).

## Compile check: the recipe's Dockerfile on `main` + only these two patches

The recipe's `docker/Dockerfile` builds `TF_REPO`@`TF_REF` and applies `patches/engine/*.patch` in filename order with
`git apply --check && git apply`. Use a scratch copy of the recipe repo as the build context so its own 106 patches are
not applied (they target f350cd3e and would fail on 8094ec96):

```bash
# on a DGX Spark (ZIG_CPU=native), from a scratch copy of the recipe repo
RECIPE=/path/to/recipe/repo          # the one whose docker/Dockerfile this is
CTX=$(mktemp -d) && cp -r "$RECIPE/docker" "$CTX/" && mkdir -p "$CTX/patches/engine"
cp /path/to/roce-any-world-patches/0001-*.patch /path/to/roce-any-world-patches/0002-*.patch "$CTX/patches/engine/"
cd "$CTX" && DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
  --build-arg TF_REPO=https://github.com/ashhart/TensorFold.git \
  --build-arg TF_REF=8094ec962ef03fb43bb879c2dde6e71869869e82 \
  --build-arg ZIG_VERSION=0.17.0 \
  --build-arg ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8 \
  --build-arg SM=121 --build-arg TOOLS=1 --build-arg PATCHES_HASH=roce-any-world-$(git -C "$RECIPE" rev-parse --short HEAD) \
  -t tensorfold-glm53-spark:roce-any-world-8094ec96 .
```

Notes: `TF_REF` must be the full 40-hex sha (the Dockerfile fetches by object id and checks equality only for 40-hex).
The build stage prints `apply 0001-...` / `apply 0002-...` and `2 engine patch(es) applied`, then `zig build install
native grammar` is the compile check (`TOOLS=1` compiles `tf-cuda-test` and so `roce_gather_tests.zig` too). The image
carries no compiler, so `zig build test` is a separate step: in the `build` stage (`--target build`) run
`docker run --rm --entrypoint zig <img-build> build test` after also adding the host test step for `glm_dflash_host`.
`main`'s `.zig-version` must still read `0.17.0` (the Dockerfile checks; `build.zig.zon` says `minimum_zig_version
0.17.0`). The ZIG_SHA256 above is the recipe's for `zig-aarch64-linux-0.17.0.tar.xz`. `main` may need other build
args the recipe never passed at f350cd3e (new kernels, new `-D` options): the first failure names them.

Then, the microbench and the model gates exactly as `docs/speed/candidates/E1-E2-notes.md` in the recipe lays them
out, with the one change to GO-C above.
