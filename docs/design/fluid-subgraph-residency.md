# Fluid Subgraph Residency

Fluid Subgraph Residency (FSR) is TensorFold's implementation of proven techniques for running a mixture-of-experts model larger than memory. It keeps the workload's live expert subgraph in a bounded resident cache and treats the unchanged checkpoint as cold storage. Its goal is resident-class exact serial decode when routing locality permits it. This is a goal to qualify per workload and memory budget, not a speed guarantee.

This design covers Tier-E, the exact path, on Flash Next with Metal. The reference is the same native engine, checkpoint, backend, precision, request and settings with resident weights. Moving expert bytes changes their addresses, never their values, arithmetic, routing or traversal. Peer RAM and other families or backends are separate extensions.

## Physical limits

Storage bandwidth and memory bandwidth are different resources. A cached file read measures memory reuse, not SSD throughput. Qualification must bypass the operating system's page cache and report the effective read bandwidth on the tested hardware. Historical measurements are not constants for another device.

Let B be cold expert bytes read per generated token and S be measured sustained SSD bytes per second. With B greater than zero, storage alone bounds single-stream throughput by S / B tokens per second. Read latency, compute and contention can lower that bound. Prefetch can overlap time with compute but cannot remove mandatory bytes or exceed the sustained bandwidth limit.

The exact path improves this limit through reuse: admit checkpoint bytes once, retain useful experts and avoid reading them again. Faster storage raises S. Changing the checkpoint representation or skipping model computation is outside this design.

## Memory ownership

Resident memory includes non-routed weights, the expert pool, its persistent address map, file metadata, bounded read or forecast storage, PLE row staging when selected, execution scratch, all admitted streams, retained prompt state and the runtime reserve. Admission counts the complete path, including startup construction peaks. A pool budget cannot justify loading the full routed weights beside the pool.

One slot owns every component needed for an expert: gate, up and down code bytes and their stored scales and biases. File offsets, component lengths, shapes, recipes and ranges are validated before use. Prepared layouts may rearrange bytes only through the resident engine's existing exact transforms.

The expert-to-slot map persists across tokens. There is no compact selected-expert weight table rebuilt per token. Slots under GPU consumption or background read ownership cannot be overwritten. A slot becomes visible only after all its component reads succeed. Errors stop the request; stale or partly filled bytes never stand in for a missing expert.

## C0: locality profiler

The profiler consumes complete routing observations, including cache hits, for a declared workload. Each observation identifies its layer, execution window and actual expert union. Decode, prompt chunks, draft windows and concurrent streams must be distinguishable. A miss-only file-load trace is insufficient to measure the live union.

Report the expert union growth curve, its byte cost, the pool budget, observed cache misses and cold bytes. A union that fits the cache makes a warm resident working set possible for the observed workload. It does not establish future routing locality or resident-class speed. A union larger than the cache establishes capacity pressure; measured misses and cold bytes determine its storage cost. Do not infer a compulsory read of the entire union on every token.

An empty, truncated or incomplete trace must not receive a fits verdict. Report forecasts separately from actual routes. Cache-policy simulation is a planning tool, not an inference or performance receipt.

## C1: persistent expert addressing

Only the streaming path reads the expert-to-slot map. The resident path keeps its existing kernels, weight addresses and launch work. Streaming operators use the resident operators' quantization recipe, accumulation, reduction order and output rounding. Slot addressing substitutes an expert's physical location without changing the logical expert ID or its place in the route combination.

A multi-row window protects the union of all its real routes. Prompt work reuses each needed expert within a chunk. Drafts and concurrent streams retain the ordinary shared engine verification and scheduling contracts.

## C2: route-driven residency

The native implementation uses load-before-execute. The real router selects experts. Cache hits acquire a lease; misses reserve evictable slots and read the checkpoint with cache bypass enabled. The layer runs only when its complete required union is available. All available slot capacity is a cache, rather than a permanently split hot and cold partition.

Cold eviction is a CLOCK policy. A hit gives a slot a second chance. Current demand is protected before eviction begins, including hits that occur later in the same union. Slots are released for reuse only after their GPU consumers complete. A failed load leaves its map entry invalid and cannot publish a token or accepted state. Cancellation joins readers and drains consumers before resources are freed.

Pool capacity and temporary allocations are admitted before execution. A configuration that cannot fund its supported active window refuses by name. It must not silently shrink top-k, change context, disable drafts or use different model arithmetic to fit.

## C3: bounded next-layer prefetch

The next layer's own router can forecast routes from the available current hidden state. The actual later router remains authoritative. Background reads reserve only slots that are safe to reuse and yield capacity to demand. Correct forecasts turn later misses into hits. Incorrect forecasts waste reads and may displace cold entries, but cannot change routes, state or tokens.

CPU writers and GPU readers have explicit ownership boundaries. Forecast inputs cannot borrow mutable scratch after it has been reused. Prefetch failure cannot expose partial bytes. Shutdown and cancellation join outstanding reads before closing descriptors or releasing buffers. Compare prefetch enabled and disabled under identical cache budgets to establish whether overlap helps.

## PLE rows on SSD

The separate PLE option retains n-gram ID generation and the resident gather formula. It reads only requested checkpoint rows and their stored metadata into bounded staging, deduplicates repeated IDs and restores request order. It must avoid materializing or pinning the full table. Shard edges, range errors, short reads and cancellation follow the same ownership rules as experts. Report its extra read traffic and latency independently.

## Qualification

Host tests cover CLOCK eviction, whole-window unions, leases, full-budget use, correct and incorrect forecasts, failure before publication, short reads, malformed metadata, truncation and reader shutdown. PLE tests compare exact row bytes across duplicates and shard boundaries. Known file-format errors are detectable; arbitrary same-size content corruption needs an independent integrity reference and must not be claimed otherwise.

A synthetic MoE device gate executes resident and streamed versions of the same native operators with fewer slots than the complete multi-layer expert set. Every logit and token must match. Test cache hits, eviction, repeated experts, forecast misses, prompt cuts, draft/plain, concurrent/solo, resumed/fresh and cancellation/reset seams. Kernel contracts include byte-equality replacements, controls and actual launch bounds.

Real checkpoint qualification uses identical complete token arrays and request settings at 16, 22, 33 and 48 GiB expert-pool budgets. Compare the complete output lengths and token sequences, not a nonempty common prefix. Record cold and warm prefill, serial decode, draft acceptance, concurrent behavior, physical footprint, device allocations, SSD bytes and timings. Use uncached reads, fresh runs and repeated alternating resident/streamed measurements. A budget on a large-memory host does not establish behavior on a smaller-memory machine.

## Prior art

These mechanisms are published techniques. EdgeMoE describes weights on storage with a memory index. MoE-Infinity and Fiddler describe expert caching based on locality or popularity and prefetch. Fate describes cross-layer gate prefetch. TensorFold re-implements these techniques in its native engine and follows its existing exactness contracts. FSR is not a novelty claim.

Slotstream is an outside comparison for SSD expert streaming and published memory/speed trade-offs. It is not the design or code source for TensorFold's implementation. Its published results are not TensorFold qualification receipts.
