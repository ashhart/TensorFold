# Check a CUDA refactor's machine code

Use this gate for a replacement that promises identical machine code, including changes to shared headers.
It compiles locally and does not execute GPU code or fetch dependencies.
Use a checkout with its pinned Zig version, CUDA's nvcc and cuobjdump, and enough disk for retained fatbins.

```sh
tools/zig/sass_compare.sh BASE_TREE NEW_TREE --repo . --output sass-proof --zig /path/to/pinned/zig
```

Each input is a local Git revision or a checkout directory.
Directory inputs include uncommitted source bytes; the receipt hashes the snapshot as well as recording its HEAD.
Choose the base before the complete refactor and the final fixed head, rather than an intermediate broken tree.
The output directory must be new; the runner retains snapshots, logs, commands, generated sources, fatbins and dumps.
Override `--nvcc` and `--cuobjdump` together if CUDA is not in `/usr/local/cuda/bin`.
There are at most two compiler jobs, at nice priority 19; `--jobs 1` serializes them.
On a Mac, also supply `--cache-guard /path/to/cache_guard.sh` before compiler builds.

The runner selects images whose source or quoted-include closure intersects changed files, ignoring preprocessor
conditions to keep the closure conservative. Changes to global compiler flags select every catalog image.
Image flags, minimum SMs and architecture-specific targets come from each tree's build catalogs.
Qwen4's GDN and PLE sources are regenerated through the same family factories and control wrappers as production.
The snapshot's checked-in copies cannot stand in for those generated sources.
Unknown flag expressions or catalog additions and removals require a separate gate and fail this runner.

Every selected image is compiled twice on both trees for sm_80, sm_86, sm_89, sm_120 and sm_121, subject to the
image's existing minimum SM. Architecture-specific images retain the production `a` suffix where required.
Every `Function` emitted by cuobjdump is compared at every supported SM, including outlined functions,
template specializations and planted control kernels. No source-name filter removes functions from comparison.
Both instruction text and complete instruction encodings, including scheduling words, must match.
Missing architectures, empty or duplicate functions, added or removed functions, incomplete encodings and unstable
rebuilds fail. The run stops at the first failing image and keeps its per-function differences.

### nvcc 13.0.88 and 13.3.73 router-tiles instability on GB10

On GB10 (aarch64 Linux), nvcc 13.0.88 and 13.3.73 with the production build flags can emit different PTX from
unchanged `moe.cu` when address randomization is enabled. Both router-tiles entries use `common/router_tiles.cuh`;
nvcc 13.0.88 also varied `nemotron_route.cu`. Its observed `tf_moe_router_tiles` alternate register counts were
193, 199 or 200 at sm_80 and 238 at sm_86, rather than 182. Disabling randomization produced 100 identical
compute_80 compiles with nvcc 13.0.88.

With randomization enabled, nvcc 13.3.73 produced other `tf_moe_router_tiles` forms in 8 of 100 `compute_80`
compiles and 9 of 50 `compute_86` compiles on GB10. The `compute_80` alternates used 182 or 193 registers; all nine
`compute_86` alternates used 182, the common form's count, so a register count alone does not establish stability.
The same 13.3.73 wheels on x86-64 gave one form in 100 and in 50; the earlier x86-64 nvcc 13.0.88 control did too.
The instability is not 13.0-only: it appeared on GB10 with both compiler versions, not on those x86-64 controls.

The observations above are of `common/router_tiles.cuh` reading each 16-byte shared-memory row as one `uint4`.
Reading it as four 32-bit words, `tf_moe_router_tiles`, `tf_nemo_router_tiles` and `tf_ctl_moe_router_tiles_row`
each gave one PTX form in every compile measured: at `compute_80` and `compute_86` with nvcc 13.0.88 and 13.3.73
on GB10, with randomization enabled and disabled; at `compute_89`, `compute_120` and `compute_121` with 13.0.88 on
one GB10 and 13.3.73 on another, with randomization enabled; and on x86-64. Their images equal the common form's
at sm_80, 86, 89, 120 and 121 with both releases, so the records keep their register counts. This is observed for
those entries, releases and hosts, not established by construction.

An unstable rebuild is a qualification failure, not evidence of an arithmetic difference. The reported comparison
did not compare every PTX operand or run the forms against each other on a GPU. Do not change a record's common
value because one rebuild differs.

`compared-functions.tsv` and `functions.json` give the exact image, SM and emitted symbol for every comparison,
with both instruction and encoding hashes and an equality status.
The emitted symbol list is compiler-dependent; a plan is not a substitute for this list.
`summary.json` records input and generated-source hashes, compiler versions, image flags and qualification status.
`compiles/` keeps each command, nvcc log, fatbin and raw SASS dump; `diffs/` holds differing functions.

Two mandatory negative controls change the FP4 pair constant and reverse the warp-sum order in separate head
snapshots. Each must change at least one function in a stable rebuild; the receipt names the actual SMs and symbols.
Their own complete comparison lists live in `control-*/compared-functions.tsv`.
Controls do not establish model fidelity and do not replace a kernel's independent numerical contract.

Only exit 0 with `mode: "sass"` and `qualified: true` in `summary.json` is an identity qualification.
`--plan-only` lists the conservative closure without a compiler, and `--generate-only` checks production source
generation without CUDA. Both leave qualification false.
`--no-controls` and `--no-recheck` are diagnostics only and return exit 2 even when their comparisons match.
An unavailable compiler or a failed compile leaves identity unchecked and returns failure; no tree is patched to pass.
