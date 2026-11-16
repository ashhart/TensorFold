//! The planner's verdict as the lane host's (sched_host.zig): which hold a wait met, its one skip, a refusal's words.
const api = @import("engine_api");
const sched = @import("kv").sched;

/// The lane host's verdict for one the planner did not admit; `before` and `after`: its counts around the call.
pub fn held(v: sched.Verdict, before: sched.Counts, after: sched.Counts, words: *[]const u8) api.Rounds.Verdict {
    return switch (v) {
        .admit => .admit, // sched_host records the admission first
        // the planner's own counters say which hold it met: the KV pool's pages or the memory floor
        .wait => .{ .wait = if (after.pool_waits > before.pool_waits) .kv_pool else .memory_floor },
        .skip => .{ .skip = .prefill_concurrent }, // its one skip: a long prompt past the concurrent bound
        .refuse => |r| blk: {
            words.* = switch (r) {
                .too_large => "the request needs more KV pool pages than the pool has",
                .memory => "the prompt's prefill cannot fit in this server's memory now (TF_DSV41_FLOOR_HARD_GIB)",
            };
            break :blk .refuse;
        },
    };
}
