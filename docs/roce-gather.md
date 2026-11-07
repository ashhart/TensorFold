# RoCE all-gather for a group of ranks

One-shot all-gather between DGX Sparks over RoCE: the GPU side of b12x's "RoCEnante" exchange (https://github.com/local-inference-lab/b12x, b12x/comm/roce, Apache License 2.0), written for two ranks and generalised to any world whose ranks all reach each other (two on one cable; two, four or more on a switch; at most `roce.max_world` = 8). The host side (zig/src/cuda/roce.zig) owns a pinned region the GPU and the NICs reach:

```
  recv[slot][peer]       each peer's shard, written by that peer's NIC (2 slots of world - 1 of slot_bytes)
  flag[slot][peer][hca]  the sequence a peer's NIC writes after its stripe on that HCA (128 bytes apart)
  send[slot]             this rank's shard, staged here for its NICs (2 slots)
  ctrl                   {seq, nbytes of slot 0, nbytes of slot 1, error seq, missing hca, missing rank, probe tokens}
```

A rank's peers are the other ranks in rank order (rank 2's peers 0, 1, 3 are its peer numbers 0, 1, 2); every rank opens one RC queue pair a peer an HCA, and at world 2 (one peer) every offset and every byte is the two-rank design's.

One launch: (1) every block stages its share of the shard into send[seq & 1]; (2) the last block to stage publishes the slot's size and rings ctrl.seq after a system fence, and the proxy thread RDMA-writes the shard to every peer, each copy on this rank's queue pair toward that peer, and then a flag per HCA stripe on the same queue pair (so a flag never lands before its stripe); (3) one thread a (peer, HCA) waits for that stripe's flag, (world - 1) x HCAs of them; (4) every block writes every shard into the output slot of its rank (the layout NCCL's all-gather gives); (5) the last block to finish advances the device epoch, so a CUDA graph replays the exchange with new sequences. A wait past the spin limit poisons the exchange: later launches return at once and the host reports the error, naming the HCA and the rank whose stripe never came.

## Host side

Small all-gathers between two ranks as one-shot RDMA writes over RoCE: b12x's "RoCEnante" exchange (https://github.com/local-inference-lab/b12x, b12x/comm/roce, Apache License 2.0), written for two ranks: the kernel (kernels/cuda/roce_gather.cu) stages this rank's shard in pinned host memory and rings a doorbell; a proxy thread here RDMA-writes it to the peer, striped over the HCAs, each stripe followed by its sequence flag on the same queue pair; the kernel waits for the peer's flags and copies both shards out. A decode step's gathers are a few rows of partials. On two GB10s with one ConnectX-7 cable (NCCL 2.30.7, `tf-cuda-test roce-gather`, 2,000 calls back to back) this path took 10.4 us a call at 4 KiB a rank against NCCL's 12.3, 11.6 against 13.5 at 8 KiB, 17.4 against 20.5 at 32 KiB, and 17.4 against 51.5 at 64 KiB. libibverbs is opened at run time (as the driver and NCCL are): without it, or without an active HCA, the caller keeps NCCL.

## Setup and use

`tp_group.Group.openRoce` brings the path up on the group: each rank opens its HCAs and a queue pair a peer an HCA, and what each brings (its region, keys, queue pairs and the largest shard it would send) crosses the group's own NCCL communicator (an all-gather), as does each rank's verdict after connecting. Every rank gets the path or none does (`TF_ROCE=0` on any rank keeps NCCL). Then a connect probe: every rank RDMA-writes a token to every peer on every HCA and waits for its writes to complete and for every peer's token; a rank with no RDMA path to another on the same HCA (a ring of four without a switch, whose far ranks share no cable) fails this, and every rank keeps NCCL, with one warning naming the rank, rather than a kernel spinning at the first gather. A rank's HCA h writes to every peer's HCA h, so the fabric must pair the same HCA index across every rank (a switch does; `fabric/tp`'s multirank wire, which a GLM rank borrows with `TF_GLM_ROCE_MULTI=1`, is the path for a ring's per-pair ports). From then on `Group.allGather` sends a gather of up to the smallest rank's cap (512 KiB by default) this way and leaves larger ones to NCCL; the size alone decides, so every rank always takes the same path. A rank whose buffers are not 16-byte aligned gets `error.Misaligned` instead of falling back alone. Liveness is `tp_watch`'s, as for every group.

A failed exchange writes nothing into the gathered rows. Call `Group.check` after any synchronize whose result reads a gather: it returns the error the GPU or the proxy recorded.

`tf-cuda-test roce-gather <rank> [world]` runs on every rank (NCCL_COMM_ID set; world 2 by default): every size up to the cap and past it against NCCL's all-gather, byte for byte, a captured graph replayed, 3,000 mixed-size calls (16, 32 and 64 KiB a rank with tiny ones between) each against NCCL's rows, then each path's time a call at 4 to 64 KiB. Its last line, `RESULT roce-gather world N rank R: 32 KiB a rank RoCE X us a call (GO at most 25.0), 3000 mixed-size calls, M mismatch(es) (GO 0)`, is the gate a recipe reads before turning the path on at four ranks. The two-rank timings above stand; four ranks on a switch are to be measured with it.
