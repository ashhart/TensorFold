//! The default-off exact-cache owners initialized beside the unchanged Qwen CUDA host buffers.
const std = @import("std");
const Cuda = @import("lanes.zig").Cuda;
const Engine = @import("engine.zig").Engine;
const c = @import("shape.zig");
const st = @import("state.zig");
const exact = @import("exact_prompt_cache.zig");
const exact_reuse = @import("exact_seq_reuse.zig");
const draft_reuse = @import("exact_draft_reuse.zig");
const dr = @import("draft.zig");

pub fn init(gpa: std.mem.Allocator, e: *Engine, streams: u32) !Cuda {
    const env = std.c.getenv("TF_PR608_EXACT_PROMPT_CACHE");
    const on = try exact.policy.enabled(if (env) |v| std.mem.span(v) else null);
    var cache = exact.Cache.init(gpa, on);
    cache.logits_enabled = try exact.logits_policy.configured(on);
    cache.post_first.enabled = try @import("exact_post_first.zig").configured(on);
    return .{
        .gpa = gpa,
        .e = e,
        .streams = streams,
        .exact_cache = cache,
        .reset_skip = .{ .enabled = try exact_reuse.reset_skip.configured(on) },
        .seq_pool = exact_reuse.Storage.init(try exact_reuse.policy.configured(on), streams),
        .draft_pool = draft_reuse.Storage.init(try draft_reuse.policy.configured(on), streams, dr.Context.bytes()),
        .logits = try gpa.alloc(u16, c.vocab),
        .cand_values = try gpa.alloc(f32, st.round_rows * st.max_candidates),
        .cand_ids = try gpa.alloc(i64, st.round_rows * st.max_candidates),
        .values = try gpa.alloc(f64, st.max_candidates),
    };
}
