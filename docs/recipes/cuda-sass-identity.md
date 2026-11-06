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
