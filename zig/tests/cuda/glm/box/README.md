# The GLM-5.3-Flash CUDA box

GLM-5.3-Flash runs on two ranks across two machines (each GB10 holds one rank), so its capture
extends [nemotron's box](../../nemotron/box/README.md) with the peer orchestration: rank 0 (the
driver) runs on this machine, rank 1 on `TF_PEER` over ssh; the NCCL bootstrap crosses the LAN, so
the containers run with `--network host`. Both ranks install the recorder and write their own
`launches.json`; the script merges both ranks' Triton caches and packs one kernel set
(`aot.json` + `cubins/`) via `tools/zig/aot_pack.py`.

## Layout

One work dir (`TF_NEMO`) holds the pieces: `src/` is a checkout of this repository that **also
carries the Python engine's `src/` tree** — the 1.0.0 split moved the Python line to the
`python-0.6` branch, and the capture imports the engine from `PYTHONPATH=/tensorfold/src`. A union
checkout (this tree plus python-0.6's `src/`, with a `TREE` file naming both pins) satisfies both
mounts. `out/` takes each run's logs and artifacts (per rank), `aot/` the Triton caches, extension
builds and the packed kernel set.

## Variables

`TF_NEMO` the work dir; `TF_MODEL_ROOT` the HF cache model directory (the `models--org--name/`
root — its `snapshots/<rev>/` holds the checkpoint; the root is mounted so the snapshot's relative
symlinks into `blobs/` resolve inside the container); `TF_PEER` the second machine's ssh target
(empty: single-rank run on this machine); `TF_MASTER`/`TF_PORT` the NCCL bootstrap endpoint
(defaults `127.0.0.1`/`29561` — for two machines, rank 0's address); `TF_ZIG_IMAGE` the container,
default `nvcr.io/nvidia/pytorch:26.07-py3`; `TF_JOURNAL` a JSON-lines file for START/END lines;
optional `TF_WHO`.

## Run

```sh
flock <GPU lock> bash -u capture.sh RUN --record [capture.py options]
```

Preflight (both machines) fails while any container or GPU compute app is present — every GPU step
assumes the caller holds the GPU lock. Rank 1 starts first and is torn down after rank 0 exits; its
`follow()` ends with the pair's connection close, which the capture catches and records through.

The packed set feeds the Zig engine's replay the way nemotron's does (`--kernels DIR` /
`TENSORFOLD_CUDA_KERNELS`), with GLM's layer graph and state layout supplied by the family.
