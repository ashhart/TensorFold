# Optional CUDA decode-weight prefetch

Qwen3.5/3.6 dense and MoE, and Nemotron, can prefetch resident decode weights into L2 while independent work runs. Both families default off on every GPU. This is a port of @Bizuayeu's GLM mechanism in #1580; GLM keeps its existing settings and launches.

Set the flag before starting a fresh server. The extra stream, events, modules, table allocation and launches exist only with the flag enabled. Prompt chunks skip these hooks. A decode window forks the read-only prefetch on its side stream and rejoins before the vocabulary head; captured graphs contain the same dependencies. Closing the engine drains the work before releasing weights.

| Family | Flag | Site budget | CTAs | Paced rate |
| --- | --- | --- | --- | --- |
| Qwen3.5/3.6 | `TF_QWEN_L2PF=1` | `TF_QWEN_L2PF_MB=12` | `TF_QWEN_L2PF_CTAS=2` | `TF_QWEN_L2PF_PACE_GBPS=150` |
| Nemotron | `TF_NEMOTRON_L2PF=1` | `TF_NEMOTRON_L2PF_MB=12` | `TF_NEMOTRON_L2PF_CTAS=2` | `TF_NEMOTRON_L2PF_PACE_GBPS=150` |

An unset flag or explicit `0` leaves the feature off and ignores its subordinate knobs. Other flag values refuse by name. Budgets and CTA counts are positive integers; a pace of zero selects the unpaced kernel. The enabled startup line names the family, budget, CTA count and rate. Changing the environment of an already running process does not rebuild its tables or graphs.

Qwen prefetches the current GDN/attention output projection, dense gate/up or MoE router, and the next layer's input projections. Nemotron prefetches Mamba/attention output projections and the next block's input projection or router. It retains its separate shared-expert stream. Neither family scans all expert weights. Sources are bounded by their actual weight allocations, with interior packed views clipped and each site's final range truncated to its budget in 16-byte multiples. Missing or misaligned resident ranges refuse rather than reading arbitrary memory.

`zig build test-l2-prefetch-host compile-l2-prefetch-host` checks settings, resident ranges, exact launch arguments, capture ordering and partial-load cleanup without a GPU. The port reuses the unchanged `l2pf` and `l2pace` images and their nvcc 13.0 launch records. Physical token equivalence, graph replay and performance still require GPU qualification. Keep these family defaults off until matched on/off device cells qualify.
