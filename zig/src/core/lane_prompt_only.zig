//! Prompt-only jobs (an embedding or a readout): what a lane host refuses before queueing, and how one completes.
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;

/// A request this host cannot serve: its kind against the engine's, or a prompt-only kind without its hooks.
pub fn refuses(h: *const LaneHost, request: *const api.Request) bool {
    if ((request.embedding != null) != (h.info_.embeddings != null)) return true;
    if ((request.readout != null) != (h.info_.readouts != null)) return true;
    if (!request.promptOnly()) return false;
    const v = h.core.backend.vtable;
    if (h.rounds != null or v.prefill_step == null) return true; // a family round planner has no completion for it
    return if (request.embedding != null) v.embedding == null else v.readout_finish == null;
}

/// The filled prompt's answer delivered, the backend's state released, the job finished.
pub fn complete(h: *LaneHost, job: *LaneHost.Job) bool {
    if (job.request.embedding != null) return @import("lane_embeddings.zig").complete(h, job);
    return @import("lane_readouts.zig").complete(h, job);
}
