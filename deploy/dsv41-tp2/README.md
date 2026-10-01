# tensorfold-dsv41-TP2

TensorFold serving DeepSeek-V4.1-Flash EXL3 2.9bpw at TP=2 across two DGX Sparks, as a docker-compose project.
It is a drop-in for the vLLM recipe (`../deepseek41flash-exl3-TP2`): same port (8888), same model id
(`DeepSeek-v4.1-Flash-EXL3`) and the same bearer API key, so clients and the frpc tunnel need no change.
The two cannot run at once (each takes ~100 GiB a rank, both bind :8888); `make swap-in` / `make swap-out`.

| file | |
|---|---|
| `Dockerfile` | NGC PyTorch 26.07 + xgrammar / transformers + a snapshot of the TensorFold source (`build/tensorfold`) |
| `docker-compose.yaml` | one service for both ranks; `rank0.env` (head) / `rank1.env` (worker) set the rank and its NICs |
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
for `MEM_READY_GIB` free on both nodes, finds each node's RoCE v2 GID index, starts the worker, then the head,
and waits for `/health`.

The containers are not restarted on their own: one rank coming back alone would wait at rendezvous for a peer
that is not coming. Start and stop both with `make`.

## Timings (2026-10-02)

- first start after an image build with an empty cache: 5m39s (CUDA extensions build into `CACHE_DIR`)
- later starts: 2m22s from `make restart` to serving (83 s load + warm-up and graph capture)
- `PARALLEL=8 CONTEXT=131072`: 6.9 GiB left after warm-up, 3.0 GiB of kept prompt states;
  8 clients 85 tok/s aggregate, concurrent replies identical to sequential ones

## Updating

Sync the new source to `SRC` on the head, then `make image && make restart`. The image tag is a hash of the
Dockerfile and the source snapshot; `docker image ls tensorfold-dsv41` lists earlier builds for a rollback
(`TF_IMAGE=tensorfold-dsv41:<rev>` in `.env`).
