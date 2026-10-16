//! A request's sampled rule as sample.cu reads it (the Metal engine's draws), and the greedy head draw's _keyed rule.

const std = @import("std");
const cuda = @import("cuda");
const Keyed = @import("cuda_triton.zig").Keyed;

pub const Sampling = @import("lanes").Sampling;

/// sample.cu's shared pieces (cuda/sample.zig): candidates beyond top_k, the most a program ranks, the count.
pub const margin = cuda.sample.margin;
pub const max_candidates = cuda.sample.max_candidates;
pub const count = cuda.sample.count;

/// The request's rule as the engine runs it: null (or temperature 0) decodes greedily.
pub fn check(s: ?Sampling) ?Sampling {
    const x = s orelse return null;
    return if (x.temperature > 0) x else null;
}

/// The greedy head draw: the top 20 ranked, the argmax and its share written (captured _keyed).
pub const greedy_draft: Keyed = .{ .k = 20, .greedy = true };

pub const Rule = cuda.sample.Rule;

/// The rule the Metal engine's sample kernels read for `s` (lanes.gpu_rule), in sample.cu's layout.
pub fn rule(s: Sampling) Rule {
    return Rule.from(@import("lanes").gpu_rule.rule(s));
}

test "rules as the Metal engine fills them" {
    const s: Sampling = .{ .seed = 1 << 63 | 5, .temperature = 0.5, .top_k = 0, .top_p = 0.9, .min_p = 0.05 };
    const r = rule(check(s).?);
    try std.testing.expectEqual(@as(u64, 1 << 63 | 5), r.seed);
    try std.testing.expectEqual(@as(f32, 2.0), r.inv_t);
    try std.testing.expectEqual(@as(f32, @floatCast(s.minLog())), r.min_log);
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Rule));
    try std.testing.expectEqual(@as(?Sampling, null), check(.{ .seed = 0, .temperature = 0 }));
    try std.testing.expectEqual(@as(usize, 28), count(20, 131072));
}
