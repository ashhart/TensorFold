const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const E = @import("cuda_engine.zig");
const cache_mod = @import("cuda_cache.zig");
const cuda_rope = @import("cuda_rope.zig");

pub fn contextCeiling(a: std.mem.Allocator, io: std.Io, dir: []const u8, flag: ?f64) !i64 {
    return @intCast((try cuda_rope.resolveDir(a, io, dir, flag)).ceiling);
}

pub fn cacheBytes(e: *const E.Engine, cap: usize, test_tokens: usize) !usize {
    return cacheBytesWithTaps(e, cap, test_tokens, e.n_taps);
}

pub fn cacheBytesWithTaps(e: *const E.Engine, cap: usize, test_tokens: usize, taps: usize) !usize {
    return std.math.add(usize, e.streamBytesWithTaps(cap, taps), try cache_mod.pageExtra(e.g, cap, test_tokens, e.kv8));
}

pub fn modelFacts(window: u32, verify: u32, g: anytype) lanes.Model {
    var facts = laneFacts(window, verify);
    facts.root_copies = @import("cuda_geometry.zig").dense27(g);
    return facts;
}

pub fn laneFacts(window: u32, verify: u32) lanes.Model {
    return .{ .exact_width = window, .hidden_rows = true, .batch_rows = verify, .max_streams = E.max_segs, .streams_exact = true, .copy_match = 5, .context_copies = true, .lone_cost_tile = 16 };
}

pub fn pick(sampling: ?lanes.Sampling, position: u64) E.Draw {
    const s = sampling orelse return .{ .rule = null, .position = position };
    if (s.temperature <= 0) return .{ .rule = null, .position = position };
    return .{ .rule = cuda.sample.Rule.from(lanes.gpu_rule.rule(s)), .position = position };
}
