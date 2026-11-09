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
    return cuda.sample.rule(s);
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

// Every CUDA family's rule as it was built before cuda.sample.ruleFor, kept here as the byte-equal oracle.
const old = struct {
    const lanes = @import("lanes");
    // spark25, kolibri1, dory, gemma4 (both openings), lfm2_moe, qwen3_5 pick: T > 0, then the core rule
    fn gated(s: ?Sampling) ?Rule {
        if (s) |sp| if (sp.temperature > 0) return Rule.from(lanes.gpu_rule.rule(sp));
        return null;
    }
    // qwen4 cuda_draw: `active`, then the core rule of sampling.?
    fn active(s: ?Sampling) ?Rule {
        const on = if (s) |x| x.temperature > 0 else false;
        return if (on) Rule.from(lanes.gpu_rule.rule(s.?)) else null;
    }
    // glm and minimax_m3 cuda_sampler: their own extern Rule, filled by hand
    const Hand = extern struct { seed: u64, inv_t: f32, top_p: f32, near: f32, min_log: f32, top_k: u32, pad: u32 = 0 };
    fn hand(s: Sampling) Hand {
        const inv_t: f32 = @floatCast(1.0 / @max(s.temperature, 1e-6));
        const min_log: f32 = @floatCast(s.minLog());
        const top_p: f32 = @floatCast(s.top_p);
        return .{ .seed = s.seed, .inv_t = inv_t, .top_p = top_p, .near = 20.0, .min_log = min_log, .top_k = s.top_k };
    }
};

fn sameRule(a: ?Rule, b: ?Rule) !void {
    try std.testing.expectEqual(a == null, b == null);
    if (a) |x| try std.testing.expectEqualSlices(u8, std.mem.asBytes(&x), std.mem.asBytes(&b.?));
}

test "cuda rule: each family's old rule is ruleFor's, byte for byte, over a settings grid" {
    try std.testing.expectEqual(@sizeOf(Rule), @sizeOf(old.Hand));
    try sameRule(null, cuda.sample.ruleFor(@as(?Sampling, null)));
    const inf = std.math.inf(f64);
    const seeds = [_]u64{ 0, 1, 1 << 63 | 5, 0x0123456789abcdef, std.math.maxInt(u64) };
    const temps = [_]f64{ -1, 0, 1e-9, 1e-6, 2e-6, 0.3, 0.5, 0.7, 1, 1.5, 2, 100, inf };
    const ks = [_]u32{ 0, 1, 7, 20, 64, 1024, 1025, 262144 };
    const ps = [_]f64{ 0, 0.1, 0.5, 0.9, 0.95, 1, 1.5 };
    const mins = [_]f64{ -1, 0, 1e-12, 0.05, 0.1, 0.5, 1 };
    var sampled: usize = 0;
    var greedy: usize = 0;
    for (seeds) |seed| for (temps) |t| for (ks) |k| for (ps) |p| for (mins) |m| {
        const s: Sampling = .{ .seed = seed, .temperature = t, .top_k = k, .top_p = p, .min_p = m };
        const got = cuda.sample.ruleFor(@as(?Sampling, s));
        try sameRule(old.gated(s), got);
        try sameRule(old.active(s), got);
        try sameRule(if (check(s)) |x| rule(x) else null, got);
        const h = old.hand(s);
        const r = cuda.sample.rule(s);
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&h), std.mem.asBytes(&r));
        if (got) |g| {
            try std.testing.expectEqualSlices(u8, std.mem.asBytes(&h), std.mem.asBytes(&g));
            sampled += 1;
        } else greedy += 1;
    };
    try std.testing.expectEqual(@as(usize, 5 * 11 * 8 * 7 * 7), sampled);
    try std.testing.expectEqual(@as(usize, 5 * 2 * 8 * 7 * 7), greedy);
}
