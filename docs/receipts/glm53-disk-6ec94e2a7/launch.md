# How every server and check ran

Three hosts (`host-a/b/c`), one GB10 and one rank each, in `nvcr.io/nvidia/pytorch:26.07-py3`
containers started with `--gpus all --network host --ipc host --shm-size 16g --ulimit memlock=-1
--device /dev/infiniband --memory 120g`, the checkpoint mounted read-only as `CKPT` (the snapshot
directory of `2d747d0e30eca6e3fe37ba63c471cdf9172faa83`; every file's sha256 in
`checkpoint-sha256.txt`, identical on the three hosts), and the shared folder
`DISK` mounted as `/learn` on every rank. Every start was preceded by
`sync; echo 3 > /proc/sys/vm/drop_caches` on each host. NCCL over the RoCE link
(`NCCL_SOCKET_IFNAME=<roce-if>`, `NCCL_DEBUG=WARN`).

## Servers (`learn_test.sh binC`, 2026-11-27)

Rank R (0, 1, 2), each on its own host, with `TF_GLM_RANK=R TF_GLM_WORLD=3
NCCL_COMM_ID=<link-a>:29651 TENSORFOLD_MEMORY_RESERVE_GIB=6`:

    LAUNCHER  # starts the three ranks and waits for "serving" on rank 0

    tensorfold-native serve CKPT --backend cuda --host 0.0.0.0 --port 8899 --name glm53b \
      --context 32768 --parallel 2 <PHASE FLAGS> --tp 3 --rank R --no-update-check

`<PHASE FLAGS>`: `--prompt-cache-gib 0` for the cache-off reference, `--prompt-cache-gib 4
--prompt-cache-disk /learn` for the disk turns. Drafts on (MTP, the default). No `--kv-cache`:
BF16 cache.

## Test

    python3 learn_test.py localhost 1 learn_fresh.json    # cache off, turns 1 and 2
    python3 learn_test.py localhost 2 learn_fresh.json
    python3 learn_test.py localhost 1 learn_disk.json     # disk, turn 1 (writes the state)
    # settle: wait until no block under /learn is a .part and the size holds, then restart the servers
    python3 learn_test.py localhost 2 learn_disk.json     # after the restart, resumes from disk

Two chat turns of a synthetic records conversation (23,192 and 23,307 prompt tokens), greedy,
256 tokens a reply. A pair is equal when the turn's token sha256 and content match between the
files; PASS needs turns 1 and 2 equal, the disk turn 2 `cached` > 0 and the cache-off turn 2
`cached` == 0. `logs/learn.log` is the driver log (the server's disk-tier lines included);
`learn_fresh.json` and `learn_disk.json` are the per-turn rows.
