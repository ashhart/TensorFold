# The Nemotron CUDA box

One work dir (`TF_NEMO`) holds the pieces: `src/` is a checkout of this repository (the container mounts it
at `/tensorfold`), `out/` takes each run's logs and artifacts, and `aot/` takes the Triton caches and the
packed kernel set. Every GPU step assumes the caller holds the GPU lock; capture and zrun run a preflight
that fails while other containers or compute apps are present.

Environment: `TF_NEMO` the work dir; `TF_MODEL` the checkpoint dir (capture); `TF_ZIG` the zig binary
(build); `TF_ZIG_IMAGE` the container image, default `nvcr.io/nvidia/pytorch:26.07-py3`; `TF_JOURNAL` a
JSON-lines file the scripts append START and END lines to; optional `TF_WHO`, `TF_RESIDENT`,
`TF_BUILD_STEPS`.

## 1. capture.sh: the oracle run (GPU)

`flock <GPU lock> bash -u capture.sh RUN --record [capture.py options]`, with `TF_NEMO` and `TF_MODEL` set.
It runs `zig/tests/cuda/nemotron/capture.py` inside the image against the Python engine: with `--record`
every Triton and extension launch is recorded (`launches.json`, plus dumps, weight digests and tokens);
`--bench` writes tokens and timings only. With `--record` the script then runs step 2 itself.

## 2. triton_aot_manifest.py: the manifest (host, no GPU)

`python3 -B ${TF}/src/tools/zig/triton_aot_manifest.py --cache ${TF}/aot/RUN/triton --launches ${TF}/out/RUN/launches.json --mount /aot/triton --out ${TF}/out/RUN/manifest.json`

Each launched Triton specialization maps to its cached cubin, metadata and ABI. `--mount` is the cache as
the run saw it inside the container (`TRITON_CACHE_DIR=/aot/triton`); the cache argument is the same cache
as this host sees it. capture.sh runs this step; run it by hand only to rebuild a manifest.

## 3. aot_pack.py: the kernel set (host, no GPU)

`python3 -B zig/tests/cuda/nemotron/aot_pack.py --manifest ${TF}/out/RUN/manifest.json --cache ${TF}/aot/RUN/triton --jit ${TF}/out/RUN/jit.json --out ${TF}/aot/RUN`

Reads the manifest's kernels and writes `aot.json` (launch facts and keys) beside the cubins, the input
`tensorfold run --kernels` loads. `--manifest` and `--cache` are repeatable in pairs, so several captures
pack into one kernel set.

## 4. build.sh: the Zig CUDA engine and its SASS check (GPU image, no compute)

`TF_ZIG=<zig binary> bash -u build.sh RUN CAPTURE_RUN`

Builds the engine with the image's nvcc into `${TF}/out/RUN/zig-out`, then proves each fatbin's SASS
against the Python extension's builds from the capture run's `torch_ext` (`sass_compare.py` per pair).

## 5. zrun.sh: one engine command (GPU)

`TF_JOURNAL=<file> bash -u zrun.sh RUN CMD ARGS...`

Runs any command on the GPU with journal lines, typically the `tensorfold run` built by step 4 against the
kernel set from step 3.
