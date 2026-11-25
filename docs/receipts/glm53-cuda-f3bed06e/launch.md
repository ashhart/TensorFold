# How every server and check ran

Three hosts, one GB10 and one rank each, in `nvcr.io/nvidia/pytorch:26.07-py3` containers started with
`--gpus all --network host --ipc host --shm-size 16g --ulimit memlock=-1 --device /dev/infiniband --memory 120g`,
the checkpoint mounted read-only as `CKPT` (the snapshot directory of `2d747d0e30eca6e3fe37ba63c471cdf9172faa83`).
Every start was preceded by `sync; echo 3 > /proc/sys/vm/drop_caches` on each host (unified memory: the page cache
out of the GPU's way). NCCL over the RoCE link (`NCCL_SOCKET_IFNAME` set to its interface), `NCCL_DEBUG=WARN`.

## Servers (`tools/conformance.py --binary`)

Rank R (0, 1, 2), each on its own host, with `TF_GLM_RANK=R TF_GLM_WORLD=3 NCCL_COMM_ID=<rank 0's link address>:29651
TENSORFOLD_MEMORY_RESERVE_GIB=6`:

    tensorfold-native serve CKPT --host 127.0.0.1 --port PORT --name glm53-cuda --no-update-check \
      --prompt-cache-gib G --backend cuda --context 32768 --parallel 2 --no-thinking --tp 3 --rank R

`G` is 0 for the cache-off server and 2 for the cached one (the `glm53-cuda` profile's), `PORT` the one the tool
picks. Rank 0 serves; ranks 1 and 2 follow. Drafts on (MTP + copies, the default). No `--kv-cache`: BF16.

    python3 tools/conformance.py tools/conformance_profiles/glm53-cuda.json --binary LAUNCHER --model-dir CKPT \
      --start-timeout 2400 --output records/conformance.json

`LAUNCHER` starts the three ranks as above and stops them when the tool ends it (its sha256 is the receipt's
`binary_sha256`).

## Checks (`tf-cuda-test`)

`glm-cap-check` at TP=3, rank R on its own host, with `TENSORFOLD_MEMORY_RESERVE_GIB=9`, `IDFILE` the NCCL id rank 0
writes and the others read, and dc.json `{"capacity": 8192}`:

    tf-cuda-test glm-cap-check CKPT dc.json R 3 IDFILE

On one GB10 each, no checkpoint:

    tf-cuda-test glm-ref-dsa
    tf-cuda-test kernel-contract glm mla
