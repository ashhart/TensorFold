//! sampler.Params and DraftParams: a request's keyed rule, compiled into the graphs, and its settings on the device.

const std = @import("std");
const Keyed = @import("cuda_triton.zig").Keyed;

pub const Sampling = @import("lanes").Sampling;

/// nemotron_h.cuda.DRAFT_TAU: sampled drafts are drawn at this fraction of the request's temperature.
pub const draft_tau = 0.6;
/// exact_sampling.MARGIN: candidates read beyond top_k.
pub const margin = 8;
/// The most candidates one _keyed program ranks.
pub const max_candidates = 256;

/// Candidates a draw takes from `vocab` columns.
pub fn count(k: usize, vocab: usize) usize {
    return @min(vocab, k + margin);
}

/// The request's rule as the engine runs it: null decodes greedily; top_k 0 (the whole-vocabulary nucleus) is refused.
pub fn check(s: ?Sampling) !?Sampling {
    const x = s orelse return null;
    if (!(x.temperature > 0)) return null;
    if (x.top_k == 0) return error.NucleusUnsupported;
    if (x.top_k + margin > max_candidates) return error.TooManyCandidates;
    return x;
}

/// The target's draw: null is torch.argmax; keyed rows keep top_k, cut at top_p and at min_p.
pub fn target(s: ?Sampling) ?Keyed {
    const x = s orelse return null;
    return .{ .k = x.top_k, .cut = x.top_p > 0 and x.top_p < 1, .minp = x.min_p > 0 };
}

/// A draft's draw: greedy ranks the top 20 and writes the argmax's share; sampled keeps top_k and writes its share at T.
pub fn draft(s: ?Sampling) Keyed {
    const x = s orelse return .{ .k = 20, .greedy = true };
    return .{ .k = x.top_k, .conf_t = true };
}

/// Params.seed: the request's seed in 63 bits.
pub fn seed(s: Sampling) i64 {
    return @intCast(s.seed & ((@as(u64, 1) << 63) - 1));
}

/// Params.fp: temperature, top_p, ln(min_p), temperature.
pub fn targetFp(s: Sampling) [4]f64 {
    const t = @max(s.temperature, 1e-6);
    return .{ t, s.top_p, s.minLog(), t };
}

/// DraftParams.fp: tau x temperature, no top-p, no min-p, and the request's temperature for the share.
pub fn draftFp(s: Sampling) [4]f64 {
    const t = @max(s.temperature, 1e-6);
    return .{ @max(draft_tau * t, 1e-6), 1.0, -std.math.inf(f64), t };
}

test "rules follow sampler.keyed" {
    const s: Sampling = .{ .seed = 1 << 63 | 5, .temperature = 0.7, .top_k = 40, .top_p = 1.0, .min_p = 0.05 };
    const x = (try check(s)).?;
    try std.testing.expectEqual(Keyed{ .k = 40, .minp = true }, target(x).?);
    try std.testing.expectEqual(Keyed{ .k = 40, .conf_t = true }, draft(x));
    try std.testing.expectEqual(@as(i64, 5), seed(x));
    try std.testing.expectEqual(@as(usize, 48), count(40, 131072));
    try std.testing.expectEqual(@as(f64, 0.42), draftFp(x)[0]);
    try std.testing.expectEqual(@as(?Sampling, null), try check(.{ .seed = 0, .temperature = 0 }));
    try std.testing.expectError(error.NucleusUnsupported, check(.{ .seed = 0, .top_k = 0 }));
    try std.testing.expectEqual(Keyed{ .k = 20, .greedy = true }, draft(null));
}
