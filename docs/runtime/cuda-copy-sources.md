# CUDA copy-source custody

Temporary uploads clone input into core-owned storage. A failed completion retains that storage until a successful
wait in its saved driver, context and stream proves completion. Destination buffers and parent CUDA objects still
need their own completion boundary. The source owner does not replace those guards.

Not every host-to-device copy goes through the owner. Qwen's kept prompt states (families/qwen3_5/cuda_reuse.zig)
restore from their own pinned snapshot with direct copies on the engine's stream. A restore marks the snapshot read
until a stream wait proves those copies done; dropping a marked snapshot waits once more and frees it only if that
wait succeeds, else keeps it with a warning.

Admission charges CPU source bytes, pinned buffer capacities and Node sizes before allocation. The process-wide
limit is 1 GiB and 4096 owners. It includes active reusable storage, quarantine, and owners inside disposal callbacks.
Driver allocator bookkeeping and event internals are outside the byte count; their owner count is bounded.
`CopySourceLimit` refuses new storage at the bound. It never frees a pending source to make room. Complete existing
work and retry, or stop submitting after an unrecoverable context failure. Limits are fixed in production.

The legacy allocator argument and Stage.gpa field remain source compatible. They do not back custody storage.
Quarantinable Node and CPU data allocations always use the core's process-lifetime page allocator. Pinned data is
owned by the driver until its completion and disposal callbacks return. Caller arenas may end after submission.

The graph wrappers track up to 64 simultaneous captures by driver, context, stream and capture mode across threads.
Temporary synchronous/default-stream and asynchronous uploaders refuse while a wrapper capture applies. A global
capture guards all contexts of that driver; other modes guard its saved context. A failed begin removes its scope.
A failed end keeps its guard until a successful end. Calls that bypass the graph wrappers are outside this contract.

Captured input must use stable Buffer or Stage storage owned by the graph's parent. Keep that storage through graph
and executable disposal. Mark every replay before launching it. Wait before mutating it for a later replay. After
disposing both graph objects, wait before release; failed disposal quarantines the storage. The helper cannot infer
an executable graph's reference to arbitrary host memory.

Registered busy owners remain visible throughout query, event destruction and host-release callbacks.
Driver.close waits for those callbacks, and frees every spare stage before the library goes, one a reaper parks
while close waits included. A retained nonbusy quarantine conservatively keeps the library loaded;
close must be retried after recovery. Callers serialize ordinary submissions and Driver destruction. The helper's
reaper/close tests exercise the supported concurrent disposal path, not arbitrary concurrent use of Driver.api.

The host fixtures use fake driver queues and real graph wrapper calls. CUDA behavior, GPU/model exactness, compiled
resources and performance need separate hardware qualification.
