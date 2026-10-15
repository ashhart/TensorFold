# RoCE all-gather for two ranks

One-shot all-gather between two DGX Sparks over RoCE: the GPU side of b12x's "RoCEnante" exchange (https://github.com/local-inference-lab/b12x, b12x/comm/roce, Apache License 2.0), written for two ranks. The host side (zig/src/cuda/roce.zig) owns a pinned region both GPUs and both NICs reach:

```
  recv[slot]       the peer's shard, written by the peer's NIC (2 slots of slot_bytes)
  flag[slot][hca]  the sequence the peer's NIC writes after its stripe on that HCA (128 bytes apart)
  send[slot]       this rank's shard, staged here for its NIC (2 slots)
  ctrl             {seq, nbytes of slot 0, nbytes of slot 1, error seq, missing hca}: the proxy's doorbell
```

One launch: (1) every block stages its share of the shard into send[seq & 1]; (2) the last block to stage publishes the slot's size and rings ctrl.seq after a system fence, and the proxy thread RDMA-writes the shard and then a flag per HCA stripe on the same queue pair (so a flag never lands before its stripe); (3) each block waits for every stripe's flag; (4) every block writes both shards in rank order (the layout NCCL's all-gather gives); (5) the last block to finish advances the device epoch, so a CUDA graph replays the exchange with new sequences. A wait past the spin limit poisons the exchange: later launches return at once and the host reports the error.

## Host side

Small all-gathers between two ranks as one-shot RDMA writes over RoCE: b12x's "RoCEnante" exchange (https://github.com/local-inference-lab/b12x, b12x/comm/roce, Apache License 2.0), written for two ranks: the kernel (kernels/cuda/roce_gather.cu) stages this rank's shard in pinned host memory and rings a doorbell; a proxy thread here RDMA-writes it to the peer, striped over the HCAs, each stripe followed by its sequence flag on the same queue pair; the kernel waits for the peer's flags and copies both shards out. A decode step's gathers are a few rows of partials, where NCCL spends tens of microseconds a call in fixed costs; this path takes about ten. libibverbs is opened at run time (as the driver and NCCL are): without it, or without an active HCA, the caller keeps NCCL.

## Control link

Two ranks over the hosts' network: a TCP control link (rank 0 listens, rank 1 connects; framed messages carry requests and per-round decisions) and the NCCL communicator it brings up (rank 0 makes the unique id and sends it).
