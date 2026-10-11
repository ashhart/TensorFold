# Tensor parallelism across processes

`zig/src/fabric/tp` runs tensor parallelism with one GPU per process, either on one host or on two hosts over RoCE.
`TF_COMM_BACKEND` picks the transport for small exchanges: `nccl` (the default), `mailbox` (two GPUs on one host)
or `roce` (two hosts). NCCL carries everything larger, so it must work on its own as well.

## Building `tf-tp-test`

The `mailbox` and `roce` backends need the mailbox kernel (`mailbox.cu`) built into the binary:

```sh
zig build tf-tp-test -Doptimize=ReleaseSafe -Dtp-nvcc=/usr/local/cuda/bin/nvcc
# or embed a fatbin built elsewhere:
zig build tf-tp-test -Doptimize=ReleaseSafe -Dtp-fatbin=/abs/path/tp_mailbox.fatbin
```

A build without `-Dtp-nvcc` or `-Dtp-fatbin` still runs `nccl` and `roce-host`, which needs no GPU.
In that build the `mailbox` and `roce` suites fail at load with `NoKernel` (an engine falls back to NCCL instead).
The `INFO` line `tf-tp-test` prints at start shows the embedded kernel's size; `0 B` means none was built in.

## Choosing the RDMA devices

| Variable | Used by | Needed when |
| --- | --- | --- |
| `TF_TP_ROCE_HCA` | the RoCE wire | the host has more than one active RoCE port that is not cabled to the peer |
| `NCCL_IB_HCA` | NCCL | the same as above; NCCL otherwise picks its own devices |
| `NCCL_SOCKET_IFNAME` | NCCL bootstrap | the host has several IP interfaces and NCCL picks one the peer cannot reach |
| `TF_TP_MASTER` | the TCP bootstrap | always on two hosts: rank 0's address as the other rank reaches it |

`TF_TP_ROCE_HCA` is `name[:gid],name[:gid]`: up to two devices, and optionally each one's RoCE v2 GID index.
Without it the rank picks the devices itself, at most `TF_TP_ROCE_HCAS` (default 2).
A host cabled to more than one neighbour should always set it explicitly.
The two hosts may list their devices in a different order and use different names.
Each device pairs with the peer device on the same IPv4 /24.

`NCCL_IB_HCA` should name the same devices without the `:gid` suffix.
In NCCL, `:N` after a device name selects port `N`, not a GID index.

`run_hosts.sh` sets `NCCL_IB_HCA` from `TF_TP_ROCE_HCA` when only that is set.
When neither is set, it first runs `tf-tp-test hca` on both ranks: the RoCE pick, made with the peer over the bootstrap.
The same rule picks the wire's devices, and each rank prints its own as `NCCL_IB_HCA=name,...`.
`hca` needs no GPU, and it exits 1 when no port shares a /24 with the peer, so no host falls back to its first ports.
The runner logs each rank's pick and stops with that reason when `hca` fails.
Both hosts must use the same mode: if one names its ports and the other names none, only the latter runs `hca`.
It finds no peer and stops after `TF_TP_PLAN_TIMEOUT` (default 300 s) with that reason.

Two DGX Spark nodes, with the second ConnectX-7 port of each cabled to the other (both PCIe functions of that
port), and the first port cabled to a third node:

```sh
# on both hosts; rank 0's link address is 192.0.2.1
export TF_TP_ROCE_HCA=rocep1s0f1,roceP2p1s0f1
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
zig/src/fabric/tp/run_hosts.sh 0 192.0.2.1 ./zig-out/bin/tf-tp-test out-rank0   # host A
zig/src/fabric/tp/run_hosts.sh 1 192.0.2.1 ./zig-out/bin/tf-tp-test out-rank1   # host B
```

`ibv_devices` lists the RDMA devices; `ibv_devinfo -d NAME` shows a port's state and link layer.
`/sys/class/infiniband/NAME/ports/1/gids/N` and `gid_attrs/ndevs/N` show each GID and its netdev.
`ulimit -l` must allow the RoCE region to be pinned (`unlimited` is simplest).

## RoCE limits and the failed-transport check

No RDMA write is larger than 1 MiB, so on `TF_COMM_BACKEND=roce` a `TF_TP_MAILBOX_MAX_KB` above 1024 fails at boot.
`TF_TP_ROCE_CHUNKED_KB` (default 0, off) sends variable-length gathers past the slot over RoCE in slot-sized ops.
It needs a kernel built from this tree's `mailbox.cu`.
At most 2 MiB is posted and not yet completed at any time.
A rank whose transport fails moves its queue pairs to the error state and keeps its region until the process exits.

`run_transport.sh MASTER_ADDR PEER_SSH OUTDIR` checks that failure path on two hosts with `tf-tp-test failfast V
transport` and stops at the first kernel log line or HCA error counter that grows; run it with no arguments for help.
