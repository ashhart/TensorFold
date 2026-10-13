//! First-head substitution only; the core forced-token selection and every sampling operation remain unchanged.
const std = @import("std");
const lanes = @import("lanes");
const Cuda = @import("lanes.zig").Cuda;
const reuse = @import("exact_seq_reuse.zig");

pub fn first(p: *anyopaque, s: *lanes.Stream, position: u64) !u64 {
    const x: *Cuda = @ptrCast(@alignCast(p));
    errdefer {
        reuse.fail(x);
        if (x.exact_cache.logits_enabled) {
            _ = x.restored.remove(s);
            x.exact_cache.failed = true;
            x.exact_cache.abandon(s, x.e.ops()) catch |err| std.log.err("first-logits purge: {s}", .{@errorName(err)});
        }
    }
    const seq = try x.settled(s);
    if (position != seq.pos) return error.InvalidQwenPosition;
    var f = x.e.forward();
    if (x.exact_cache.logits_enabled) {
        const proven = x.seqs.get(s) == seq and x.restored.matches(s, seq, s.prompt().len);
        if (!try x.exact_cache.restoreFirstLogits(s, seq, f.ops, f.s.logits, proven)) {
            x.exact_cache.stats.first_logits_fallbacks += 1;
            try f.head(seq);
            _ = try x.exact_cache.captureFirstLogits(s, seq, f.ops, f.s.logits);
        }
    } else try f.head(seq);
    var out: [1]u32 = undefined;
    try x.drawRows(&.{s.sampling}, &.{position}, &out);
    return out[0];
}
