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

A build without `-Dtp-nvcc` or `-Dtp-fatbin` still runs `nccl`, `roce-host` and `tcp-host`; the last two need no GPU.
In that build the `mailbox` and `roce` suites fail at load with `NoKernel` (an engine falls back to NCCL instead).
The `INFO` line `tf-tp-test` prints at start shows the embedded kernel's size; `0 B` means none was built in.

## Choosing the RDMA devices

| Variable | Used by | Needed when |
| --- | --- | --- |
| `TF_TP_ROCE_HCA` | the RoCE wire | the host has more than one active RoCE port that is not cabled to the peer |
| `NCCL_IB_HCA` | NCCL | the same as above; NCCL otherwise picks its own devices |
| `NCCL_SOCKET_IFNAME` | NCCL bootstrap | the host has several IP interfaces and NCCL picks one the peer cannot reach |
| `TF_TP_MASTER` | the TCP bootstrap | always on two hosts: rank 0's address as the other rank reaches it |
| `TF_RANK_SECRET_FILE` | the TCP bootstrap | always on two hosts: the same owner-only key file on each, proven on every connection |

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

## TCP between two hosts without RDMA

`tcp.zig` is the `Collective` contract over one stream socket a pair of ranks, `Kind.tcp`, for hosts with no RDMA
(two AMD boxes on USB4 networking, or plain Ethernet). It moves host memory; a device side that stages through it comes
in a later pull request. It is not a `TF_COMM_BACKEND` value yet: only its tests and `tf-tp-test tcp-host` open it.

- Every frame starts with a header both ranks check: call, element type, operation, count and the link's frame
  number; a variable-length gather sends every rank's lengths first in its payload, and those are compared too. A
  call one rank makes differently fails as `Invalid`, or `Aborted` where the other rank refused it first.
  Exchange partners that disagree (rank 0 trades with 1 while 1 trades with 2) wait on frames that never come and fail
  as `Timeout`.
- A frame that has not arrived by the call's deadline (`Options.timeout_ns`) is `Timeout`, and the communicator is
  closed. The transport never resends a frame, so a late one can never land in another call or reorder a sum. The
  kernel may still retransmit a lost segment, in order, within the deadline; its retransmit floor is a host setting
  (`ip route ... rto_min 8ms`), not a transport one.
- A frame that cannot be sent fails its call (`PeerFailed`, or what the peer's frame says), and no frame starts on a
  link where one was half sent.
- `abort`, from any thread, wakes this rank's waits and queues an abort frame on every link with no frame half sent
  (waiting at most 50 ms for room). A rank waiting on any peer hears it as `Aborted`, even in a call that does not
  involve the rank that aborted, when the abort frame is the next thing on that link; a frame the aborting rank had
  already queued for a later call comes first, and then the call ends on its own links or at its deadline. On a link
  where a frame was half sent no abort frame can go, and the peer sees `PeerFailed`, as it does from a peer that hangs
  up without one. A rank whose peer timed out after its frame was sent may finish that call and fail on its next.
- Sums add the ranks in rank order (`reference.reduce`), so every rank's bytes equal the reference's.

`tf-tp-test tcp-host`, which needs no GPU, checks it between two hosts over the bootstrap's plan link (rank 0 listens on
one port, `TF_TP_PLAN_PORT`, else `TF_TP_PORT` + 7): all-gathers, fp32
and bf16 sums, exchanges, send and recv and variable-length gathers from 1 B to 4 MiB, bit for bit, then each size's
median and p90 time (`gb_s_p50` is shard bytes over the median). Every result names the link from `TF_TP_LINK`, so a
TCP number is never read as a RoCE one:

```sh
# both hosts; rank 0's link address is 192.0.2.1
export TF_TP_WORLD=2 TF_TP_MASTER=192.0.2.1 TF_TP_LINK="USB4 networking, cable 1 of 2, MTU 1500"
TF_TP_RANK=0 ./zig-out/bin/tf-tp-test tcp-host   # host A
TF_TP_RANK=1 ./zig-out/bin/tf-tp-test tcp-host   # host B
```

## RoCE limits and the failed-transport check

No RDMA write is larger than 1 MiB, so on `TF_COMM_BACKEND=roce` a `TF_TP_MAILBOX_MAX_KB` above 1024 fails at boot.
`TF_TP_ROCE_CHUNKED_KB` (default 0, off) sends variable-length gathers past the slot over RoCE in slot-sized ops.
It needs a kernel built from this tree's `mailbox.cu`.
At most 2 MiB is posted and not yet completed at any time.
A rank whose transport fails moves its queue pairs to the error state and keeps its region until the process exits.

`run_transport.sh MASTER_ADDR PEER_SSH OUTDIR` checks that failure path on two hosts with `tf-tp-test failfast V
transport` and stops at the first kernel log line or HCA error counter that grows; run it with no arguments for help.

## Experimental four-rank ring

The multirank wire supports gathers for three through eight ranks. Its reductions and exchange use NCCL.
For exactly four ranks, `TF_TP_ROCE_RELAY=1` opts into a ring with one HCA to each neighbour and no opposite link.
Name both neighbour HCAs explicitly. The rank order follows the physical ring. All four ranks must opt in;
the connection table and the existing NCCL byte probe must agree before the wire is retained.
Default mesh and two- and three-rank paths keep their existing protocol.

Each neighbour receives both halves and forwards one to the opposite rank. Consumers wait for six flags.
The proxy completes each forward before posting the other direction and posts sequence s only after consuming
s-2 and receiving every rank's s-1. Variable-length shards carry their padded length beside each half flag.
A producer must use the existing sequential stream contract: its preceding consumer completes before staging
another collective. One proxy owns every QP. Wait failures report the source rank and sequence.

`zig build test-roce-multi-host` runs 100,000 random collectives on each of the fake-verbs and shared-memory
wires, checks every payload word, and covers held readers, full slot boundaries and sequence wrap.
`zig build test-roce-relay-controls-host` runs the focused custody, fault and arithmetic controls.
These are host protocol checks. Four-rank CUDA/RDMA execution and performance remain unqualified.
