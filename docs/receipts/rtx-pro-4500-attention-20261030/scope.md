# RTX PRO 4500 decode tuning

The two changes are independent and preserve the existing checkpoint arithmetic and serving features.
All speed figures describe the measured RTX PRO 4500 Blackwell (SM120, 82 SMs); they are not predictions for a 5090.

The affine tile changes the dense 27B MLP projection's column tile from 64 to 128, with eight warps instead of four,
only for N=17408, K=5120, one K slice and one to sixteen rows on the measured device class.
Other shapes and devices retain their dispatch. The grouped projection and vocabulary head paths are unchanged.

Attention uses two MMA row tiles and 128 threads instead of four tiles and 256 threads.
It applies only to the dense 27B geometry, FP8 KV, SM120/82 SMs and one segment.
One-row decode keeps this tile at every supported context. Wider windows of up to sixteen rows select it only
when 8192 <= cached tokens <65536. The configured 131072 context is unchanged; wider windows outside that interval
keep the original tile. A separate graph-cache bit prevents replaying a graph with the other launch geometry.

The GDN preparation/convolution-state-commit fusion is excluded. Two bracketed screening comparisons found only
+0.21%/+0.29% plain gains at 32K/8K, alongside -0.88%/-1.31% DFlash2 results. Background CPU compilation ran during
that screening, so the tiny plain differences are not CPU-idle qualification. Its microbenchmark win did not
establish a serving benefit worth submitting.

Historical tuning measurements used a common private zero-reserve compatibility overlay in both arms. The PRs
retain upstream's reserve policy. Clean-source validation uses the upstream-supported 0.5 GiB reserve in both arms.
These settings are labeled separately rather than pooling their measurements.

No stored weights, quantization, activation/accumulator precision, context limit, vision support, reasoning support,
tool/schema behavior, drafting policy, sampling settings, or scheduler/concurrency limit is changed.
Logprobs were already refused by the measured baseline; matching that refusal is not a new support claim.

The available hardware is one RTX PRO 4500 and a 32 GiB-class Linux host, with bounded test containers. A container
memory budget is not a measured minimum host-RAM requirement. Compiler records for SM121 do not qualify a device
run there. Metal, HIP, other NVIDIA GPUs, and other checkpoints require their own evidence.

The unmodified baseline returned QwenContextFull after four tokens for a 131056-token input with 16 output tokens in the configured 131072 window. The failed edge case is retained. Cold comparison uses 130816 input tokens, leaving 256 positions for output and speculative lookahead, with context 131072 unchanged in every arm. This does not qualify the rejected 131056 case.
