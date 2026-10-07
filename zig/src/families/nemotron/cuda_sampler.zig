//! A request's sampled rule as sample.cu reads it (the Metal engine's draws), and the greedy head draw's _keyed rule.

const std = @import("std");
const Keyed = @import("cuda_triton.zig").Keyed;

pub const Sampling = @import("lanes").Sampling;

/// exact_sampling.MARGIN: candidates read beyond top_k.
pub const margin = 8;
/// The most candidates one _keyed program ranks.
pub const max_candidates = 256;

/// Candidates a draw takes from `vocab` columns.
pub fn count(k: usize, vocab: usize) usize {
    return @min(vocab, k + margin);
}

/// The request's rule as the engine runs it: null (or temperature 0) decodes greedily.
pub fn check(s: ?Sampling) ?Sampling {
    const x = s orelse return null;
    return if (x.temperature > 0) x else null;
}

/// The greedy head draw: the top 20 ranked, the argmax and its share written (captured _keyed).
pub const greedy_draft: Keyed = .{ .k = 20, .greedy = true };

/// sample.cu's Rule: the seed, 1 / T, top_p, the candidates' near window, ln(min_p) and top_k (0: off).
pub const Rule = extern struct { seed: u64, inv_t: f32, top_p: f32, near: f32, min_log: f32, top_k: u32, pad: u32 = 0 };

/// The rule the Metal engine's sample kernels read for `s`, as its host fills cfg.
pub fn rule(s: Sampling) Rule {
    return .{ .seed = s.seed, .inv_t = @floatCast(1.0 / @max(s.temperature, 1e-6)), .top_p = @floatCast(s.top_p), .near = 20.0, .min_log = @floatCast(s.minLog()), .top_k = s.top_k };
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
