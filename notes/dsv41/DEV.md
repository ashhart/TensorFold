# Dev loop: DeepSeek-V4.1-Flash at TP=2 (aiai rank 0, aiai2 rank 1)

`tools/dsv41_run2.sh ARGS` rsyncs this checkout to `~/tensorfold` on both nodes and runs
`tools/dsv41_serial_run.py` in `tf-dev` (repo at `/tf`) on both ranks. A run's fixed cost is the weight load
(~82 s from the checkpoint, ~24 s from a prepared folder) plus graph capture and warm-up; the suites below pay it once
a group instead of once a test.

## Suites and tiers

    tools/dsv41_suite2.sh quick                 # engine smoke at short context: 2 processes
    tools/dsv41_suite2.sh full                  # quick + long context and quality: 3 processes
    tools/dsv41_suite2.sh long                  # the long-context group alone
    tools/dsv41_suite2.sh step-test,resume-test # named tests (their groups)
    python3 tools/dsv41_suite.py fresh full     # every test as a single dsv41_run2.sh command

A test is one of dsv41_serial_run.py's modes with fixed arguments (`tools/dsv41_suite.py`, `TESTS`). Tests that need
the same construction share a group: one process, one weight load, and `fresh_state` between tests (every cache, ring
and staging ring zeroed in place, each slot back on its construction extent with no tokens, no kept states or bank,
the drafter's rings clear, the torch seed and peak stats reset; the decode graphs stay, as captured at start).

| group | construction | tests |
|---|---|---|
| `q` | `--cap 16384 --slots 2 --graph --dspark 3` | multi-test (24 steps), step-test (8000 tokens, drafted rounds), resume-test (3000,600), views-test (3000), chunk-prefill (`TF_CHUNK_PREFILL=1 --chunk-test 6000,1000,3000,4500`) |
| `fp8` | the same, `TF_DSV41_KV=fp8` (read at import: its own process) | fp8-multi-test, fp8-resume-test (identity spot checks of the fp8 caches) |
| `long` | `--cap 140000 --slots 1 --graph` | decode-bench (32K, 128K), needle (32K, 128K at 10/50/90%), tf-compare (32K, 96K), prefill-bench (8K, 64K) |

Tiers: **quick** = `q` + `fp8`; **full** = quick + `long`. Each test prints
`[result] NAME PASS|FAIL|INFO|WARN|ERROR fp=<fingerprint> <seconds>s <summary>`; a group ends with a `[suite]` line
(rank 1 prints the same into its log, `~/tensorfold/serial-r1.log` on aiai2), `dsv41_suite2.sh` merges the groups
into `out/suite-<stamp>/summary.txt` and exits 1 on a FAIL or an ERROR. A test that raises (an OOM under the
allocator cap included) is an ERROR and stops its group: the ranks may be out of step after it.

PASS means: batched == alone and the verify window rows match (multi-test); whole == chunked == drafted (step-test);
COPY, NVMe bytes and TAKEOVER resumes == fresh (resume-test); every kept view equal on both ranks (views-test); last-row
logits bit-equal for every split (chunk-prefill); every needle recalled. decode-bench and prefill-bench report (INFO);
tf-compare reports, and with `--suite-baseline /tf/out/<an earlier suite-...-long.json>` FAILs when a document with
the same tokens loses more than 0.01 NLL or 1 point of top-1 (decode-bench: WARN when 10% slower). The markdown and
code documents are this tree's own files, so only unchanged ones are compared; the golden document always is.

### When quick suffices, when the long tier is required

- **quick** for changes that keep every long-context path as it was: the decode step, verify / drafted rounds, slot and
  extent handling, kept states, tools and serving code, kernels whose inputs are short (<= 16K positions).
- **full** (the long tier) for anything touching long-context paths: KV formats and their quantization (fp4 / fp8 /
  bf16, `TF_DSV41_KV*`, `IQ_FP4`, `SWA_FP8`, `COMP_BF16`), the indexer and top-k selection (`topk.py`, `FAST_TOPK`,
  tie keys, widths), compressed-entry pools and the shared pool, prefill (chunk plans, the bounded tail, Engram reads
  in prompt chunks), attention kernels (`mqa_fp4`, Triton MQA), and before a deploy. Run the long tier on the
  parent first: every `dsv41_suite2.sh` run writes `/tf/out/suite-<stamp>-<group>.json` on aiai (copied to
  `out/suite-<stamp>/`); then run the changed tree with `--suite-baseline /tf/out/suite-<stamp>-long.json`.

### A test in a suite equals the same test in a fresh process

A suite test parses the very argument list its fresh run gets (`Test.argv`: the group's construction, then the mode),
and a fresh run whose flags and mode environment are a suite test's prints the suite name in its `[result]` line. The
fingerprint covers the deterministic outputs only (tokens, logits diffs, view hashes, NLL values; never timings), so

    TF_SUITE_PROVE=1 tools/dsv41_suite2.sh quick

runs the tier, then every test again in its own process, and `python3 tools/dsv41_suite.py prove` checks that each
fresh run printed the suite's fingerprint (`out/suite-<stamp>/prove.txt`).

## Prepared weights in dev runs

The compose server's `make prepare` writes each rank's built weights to `PREPARED_DIR`
(`/home/urtho/.cache/tensorfold-prepared`, ~99 GB a node). A dev run reads them when

1. tf-dev mounts that directory at `/prepared` (`dsv41_run2.sh` checks both nodes and sets `TF_DSV41_PREPARED=/prepared`;
   `TF_DSV41_PREPARED=` builds from the checkpoint), and
2. the folder's key matches: the checkpoint files (same `/models` mount), rank and world, torch's version and the GPU
   (tf-dev and the image share `nvcr.io/nvidia/pytorch:26.07-py3`), the knobs `TF_FOLD_SHARED` / `TF_GROUPED_WO_A`
   (empty in both), whether the DSpark blocks are in, and the source of the code that builds the weights
   (`fastboot.CODE`: weights.py, reader.py, split.py, config.py, exl3 format / linear / experts, direct_read). Changes to
   serial.py, kernels.py, dspark.py and the rest do not touch it. `make prepare` builds with the DSpark blocks, so
   `--draft-weights auto` (the default) loads them whenever that folder is current, with or without `--dspark`; the
   engine ignores them until `enable_dspark`.

A key miss is not an error: the run builds from the checkpoint as before and says why (`key differs (code)`). Nothing in
a dev run writes a folder: aiai's disk holds one (79 GB free beside a 99 GB folder on 2026-10-05), and `make prepare`
prunes the old one first. To recreate tf-dev with the mount (once a node, both nodes):

    python3 tools/dsv41_tfdev_recreate.py aiai            # dry run: docker inspect saved to out/, commands printed
    python3 tools/dsv41_tfdev_recreate.py aiai --apply    # commit, rename + stop the old one, run the new one, check

It refuses while anything but `sleep infinity` runs in tf-dev, keeps the container layer (`docker commit`: the
editable install of `/tf`) and the old container (stopped, as `tf-dev-old-<stamp>`), and rolls back when the new one
cannot import tensorfold or see `/prepared`.

## OOM guard

GB10's GPU allocations are the host's memory; an OOM in a dev run has rebooted aiai three times (the host watchdog).
Three layers, defaults keeping >= 3 GiB available:

1. **Refuse to start** when MemAvailable is under `TF_MEM_MIN_START_GIB` (100; a rank's weights are ~99 GiB): another
   job holds the memory.
2. **Allocator cap** (`tools/dsv41_memguard.py`): `torch.cuda.set_per_process_memory_fraction` at MemAvailable less
   `TF_MEM_RESERVE_GIB` (3) and `TF_MEM_SLACK_GIB` (3: CUDA context, NCCL, ~40 MB of driver memory a decode graph,
   pinned staging), so torch raises `OutOfMemoryError` in the process instead of exhausting the node.
   `TF_MEM_CAP_GIB=N` sets it, `0` turns it off. Long runs near the edge (600K contexts) may need a smaller slack.
3. **Watcher** (`tools/dsv41_memwatch.sh`): `dsv41_run2.sh` starts one inside tf-dev on each node (`docker exec -d`,
   same PID namespace and user as the run) for this run's tag (`--run-tag`); it reads MemAvailable every 0.25 s and
   SIGKILLs only that run's processes below `TF_MEMWATCH_GIB` (3; `0`: no watcher), logging a `KILL` line to
   `~/tensorfold/memwatch.log` that run2.sh prints. It exits with the run (and run2.sh stops it on exit or Ctrl-C,
   killing a rank of the run still there 15 s after the other ended).

`docker update --memory` on tf-dev is not used: GB10 GPU allocations are not charged to the container's cgroup
(2026-10-05: tf-dev's `memory.current` 6.9 GB, `anon` 2.2 GB, while its rank held 108,893 MiB per nvidia-smi), so it
would bound only host-side memory and could OOM-kill a run on page cache.

## Serving the current source from the compose deployment

`make hot` in `deploy/dsv41-tp2` (see its README) serves `SRC`'s current `src/` without `make image`; `make cold`
returns to the image. Kernel (`.cu`) changes still compile once on the first start after them (in `prebuild`).
