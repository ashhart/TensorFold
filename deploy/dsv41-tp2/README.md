# tensorfold-dsv41-TP2

TensorFold serving DeepSeek-V4.1-Flash EXL3 2.9bpw at TP=2 across two DGX Sparks, as a docker-compose project.
It is a drop-in for the vLLM recipe (`../deepseek41flash-exl3-TP2`): same port (8888), same model id
(`DeepSeek-v4.1-Flash-EXL3`) and the same bearer API key, so clients and the frpc tunnel need no change.
The two cannot run at once (each takes ~100 GiB a rank, both bind :8888); `make swap-in` / `make swap-out`.

| file | |
|---|---|
| `Dockerfile` | NGC PyTorch 26.07 + xgrammar / transformers + a snapshot of the TensorFold source (`build/tensorfold`) |
| `docker-compose.yaml` | one service for both ranks; `rank0.env` (head) / `rank1.env` (worker) set the rank and its NICs |
| `docker-compose.hot.yaml` | `make hot` only: the synced source over the image's (below) |
| `.env` | shared settings: paths, port, model id, API key, `PARALLEL`, `CONTEXT` (from `.env.example`, chmod 600) |
| `Makefile` | runs on the head; mirrors this directory to the worker (`WORKER=aiai2-ib`) at the same path |

## Use

    make env          # first time: .env from .env.example, then set TF_API_KEY
    make image        # snapshot SRC (/home/urtho/tensorfold), build tensorfold-dsv41:<rev> + :latest on both nodes
    make swap-in      # vLLM down, then `make up`
    make status / logs / logs-worker / smoke
    make swap-out     # TensorFold down, vLLM up

`make up` refuses while vLLM, a dev run in `tf-dev`, or anything else holds the GPUs or :8888, while the house
memory watchdog is armed, or (with the carveout) while `nvidia_drm` modeset is off. It drops caches, waits
for `MEM_READY_GIB` free on both nodes, checks that every RoCE device has a RoCE v2 IPv4 GID (and
logs its index), starts the worker, then the head, and waits for `/health`. Nothing pins the GID index: the
index moves (a peer's reboot can leave aiai2's `roceP2p1s0f1` at 4 while `rocep1s0f1` stays at 3), so NCCL picks
it per device (`NCCL_IB_ROCE_VERSION_NUM=2`, `NCCL_IB_ADDR_FAMILY=AF_INET`, `NCCL_IB_ADDR_RANGE` 10.42.0.0/15) and
the RoCE all-gather finds it by type and address (`TF_RDMA_GID_INDEX` pins it).

The containers are not restarted on their own: one rank coming back alone would wait at rendezvous for a peer
that is not coming. Start and stop both with `make`.

## Kept prompts on NVMe

With `TF_DSV41_DISK=/kvdisk` (and `KV_DISK_DIR`, node-local NVMe, on both nodes; see `.env.example`) a kept prompt
state the shared pool evicts is written to disk (both ranks, in step) and restored into the pool when a later prompt
resumes from it: a read of ~45 MB for 32K tokens (fp4 KV; ~75 MB in fp8) instead of half a minute of prefill. `make down` ends the live
streams and writes every kept state first (`STOP_GRACE_S`, 150 s when the tier is on); the next start keeps the
entries both ranks hold. Entries live in a directory named by a hash of the build (source, model files, knobs,
libraries, device): a new image starts empty, and the old directories can be deleted. `TF_DSV41_DISK_GIB` (128)
bounds the space, least recently used first; `TF_DSV41_DISK_MIN` (2048) is the smallest state written;
`TF_DSV41_DISK_STAGE_MIB` (1024) of pinned memory a rank stage the copies. `/models` may share the NVMe (Engram reads).

## Watchdog

`watchdog.sh` checks the pair once a minute from a systemd user timer (`make watch-install`; the user must linger
for it to run at boot). A tick is bad when a container is not running, `/health` is not 200 (`TF_HEALTH=strict`
answers 503 for a fatal engine or one engine call running longer than `TF_STALL_S`), or a 1-token probe sent
every 10 minutes while idle fails. After `WATCH_FAILS` bad ticks (a dead container counts 2) it saves both ranks'
logs and, with `WATCH_HEAL=1`, runs `make restart` in the background, at most once every `WATCH_MIN_HEAL` seconds;
`WATCH_HEAL=0` (the default until the cluster tests below pass) only alerts. It stands down while:

- `make lease` is unexpired (`LEASE_MIN`, default 20 minutes; benchmarks take one),
- an `up` / `down` / `restart` holds the lock (`~/.local/state/tensorfold-dsv41/lock`),
- both containers are absent after a deliberate `make down` (the stop marker holds across reboots; `make up` clears it),
- vLLM runs on either node, or the head container is still loading (younger than `WATCH_GRACE`, 1800 s).

Both containers absent without the marker (a reboot while serving) is a bad tick: the watchdog then starts the
pair. `make watch-status` shows the timer, the counters and the last log lines; state and logs are in
`~/.local/state/tensorfold-dsv41/`.

Before setting `WATCH_HEAL=1`: check that `make up` runs from the unit (no TTY: `sudo -n` for `guard` / `memwait`
and `ssh -o BatchMode=yes aiai2-ib true` must work), then on the pair: `docker kill` the worker (healed within
~2 min); `kill -STOP` rank 1's python (`/health` still answers, `stalled` turns true, healed); `make down` and a
reboot (stands down); `make lease && make soak` (no heal); two failures within 30 minutes (the second only alerts);
the house mem-watchdog armed during a heal (the `guard` failure lands in `heal.log` and alerts).

## Bench

`make soak [MINUTES=30]`, `make stress` and `make structured` run `tools/dsv41_soak.py`, `dsv41_stress.py` and
`dsv41_structured.py` from `SRC` against the endpoint, with the API key in their environment (never on a command
line), after a watchdog lease covering the run; reports go to `results/`. The soak passes with no errors, the server
drained at the end, the 17×23 sanity answer (391), no fatal `/health`, and one `token_sha` for every repeated greedy
request. `structured --suites schemas,tools` needs `TF_DSV41_TOOL_GRAMMAR=required` on the server.

## Timings (2026-10-02)

- first start after an image build with an empty cache: 5m39s (CUDA extensions build into `CACHE_DIR`)
- later starts: 2m22s from `make restart` to serving (83 s load + warm-up and graph capture)
- `PARALLEL=8 CONTEXT=131072`: 6.9 GiB left after warm-up, 3.0 GiB of kept prompt states;
  8 clients 85 tok/s aggregate, concurrent replies identical to sequential ones

## Updating

Sync the new source to `SRC` on the head, then `make image && make restart`. The image tag is a hash of the
Dockerfile and the source snapshot; `docker image ls tensorfold-dsv41` lists earlier builds for a rollback
(`TF_IMAGE=tensorfold-dsv41:<rev>` in `.env`).

## Hot source (Python changes without an image build)

    make hot     # SRC's src/ -> build/hot/src (both nodes), down, prebuild, up with docker-compose.hot.yaml
    make cold    # back to the image's own source (down, prebuild, up)

The image installs TensorFold editable (`pip install -e /opt/tensorfold`, whose `.pth` names `/opt/tensorfold/src`),
so `docker-compose.hot.yaml` mounts `build/hot/src` read-only over `/opt/tensorfold/src` and the container imports
the synced tree; nothing else in the image changes. While `build/hot/REVISION` exists every `up`, `restart`,
`prepare` and watchdog heal uses the override (`make status` says so); without it the compose commands are exactly the
image ones. `TF_REVISION=hot-<hash>` keeps per-image caches (the decode cost curves) apart; bytecode goes to
`/root/.cache/tensorfold-pycache`.

- CUDA extensions are still built into `CACHE_DIR` and keyed by their sources: a changed `.cu` compiles once, in the
  `prebuild` step `make hot` runs before the start (not beside a loaded model); unchanged ones load from the cache.
- A change to the code that builds the weights (`fastboot.CODE`) misses the prepared folder: starts build from the
  checkpoint (~82 s instead of ~24 s) until `make cold`. `make prepare` refuses while hot (`PREPARE_HOT=1` overrides):
  one folder fits on aiai's disk and it would replace the image's.
- `make image` remains the clean path: a pinned snapshot both nodes build, prebuilt and prepared. It warns while hot
  source is still mounted.
- Measured 2026-10-05 (merged dsv41-cuda, extensions cached, prepared folders current): `make hot` 119 s from the
  command to serving (the server's own boot 89 s: weights 24 s, decode graphs 20 s, warm-up 12 s, calibration 25 s),
  `make cold` 117 s; `make image` after Python-only changes 10 s plus the same restart.
