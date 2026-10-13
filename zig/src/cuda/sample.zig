//! sample.cu's keyed draws (the Metal engine's rule) for every CUDA family: the rule, the count, the launches.
const std = @import("std");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// exact_sampling.MARGIN: candidates read beyond top_k.
pub const margin = 8;
/// The most candidates one keyed program ranks.
pub const max_candidates = 256;

/// Candidates a draw takes from `vocab` columns.
pub fn count(k: usize, vocab: usize) usize {
    return @min(vocab, k + margin);
}

/// sample.cu's Rule: the seed, 1 / T, top_p, the candidates' near window, ln(min_p) and top_k (0: off).
pub const Rule = extern struct {
    seed: u64,
    inv_t: f32,
    top_p: f32,
    near: f32,
    min_log: f32,
    top_k: u32,
    pad: u32 = 0,

    /// The kernels' layout of a rule built once (lanes.gpu_rule.rule): the same fields, in this order.
    pub fn from(r: anytype) Rule {
        return .{ .seed = r.seed, .inv_t = r.inv_t, .top_p = r.top_p, .near = r.near, .min_log = r.min_log, .top_k = r.top_k };
    }
};

test "sample.cu's Rule layout" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Rule));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Rule, "seed"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(Rule, "inv_t"));
    try std.testing.expectEqual(@as(usize, 12), @offsetOf(Rule, "top_p"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(Rule, "near"));
    try std.testing.expectEqual(@as(usize, 20), @offsetOf(Rule, "min_log"));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(Rule, "top_k"));
}

fn int(x: usize) i32 {
    return @intCast(x);
}

/// sample.cu's draw and logprob kernels, resolved from the loaded `sample` module.
pub const Draws = struct {
    draw_fn: Function,
    draw_ids_fn: Function,
    logprob_rows_fn: Function,

    pub fn resolve(mod: Module) !Draws {
        return .{ .draw_fn = try mod.function("tf_draw"), .draw_ids_fn = try mod.function("tf_draw_ids"), .logprob_rows_fn = try mod.function("tf_logprob_rows") };
    }

    /// Row r of bf16 logits drawn at position meta[0] + r + 1 + offset; columns are `ids` token ids if given.
    pub fn draw(d: *const Draws, s: Stream, logits: u64, vocab: usize, rule_ptr: u64, meta: u64, offset: usize, out: u64, rows: usize, ids: ?u64, prob: ?u64) !void {
        var a: launch_.Args = .{};
        a.add(logits);
        a.add(@as(u32, @intCast(vocab)));
        a.add(rule_ptr);
        a.add(meta);
        a.add(int(offset));
        a.add(out);
        if (ids) |x| a.add(x);
        a.add(prob orelse 0);
        const f = if (ids != null) d.draw_ids_fn else d.draw_fn;
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 1024 } }, s, &a);
    }

    /// Each bf16 logits row's logprob words at `out`: its pick in `picks`, its `k` best tokens, its rank if `rank`.
    pub fn logprobRows(d: *const Draws, s: Stream, logits: u64, vocab: usize, picks: u64, k: u8, rank: bool, out: u64, rows: usize) !void {
        var a: launch_.Args = .{};
        a.add(logits);
        a.add(@as(u32, @intCast(vocab)));
        a.add(picks);
        a.add(@as(u32, k));
        a.add(@as(u32, @intFromBool(rank)));
        a.add(out);
        try launch_.launch(d.logprob_rows_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 1024 } }, s, &a);
    }
};

test "a rule in the kernels' layout keeps every field" {
    const r = Rule.from(.{ .seed = @as(u64, 1 << 63 | 5), .inv_t = @as(f32, 2.0), .top_p = @as(f32, 0.9), .near = @as(f32, 20.0), .min_log = @as(f32, -3.0), .top_k = @as(u32, 7) });
    try std.testing.expectEqual(@as(u64, 1 << 63 | 5), r.seed);
    try std.testing.expectEqual(@as(f32, 2.0), r.inv_t);
    try std.testing.expectEqual(@as(f32, 0.9), r.top_p);
    try std.testing.expectEqual(@as(f32, 20.0), r.near);
    try std.testing.expectEqual(@as(f32, -3.0), r.min_log);
    try std.testing.expectEqual(@as(u32, 7), r.top_k);
    try std.testing.expectEqual(@as(usize, 28), count(20, 131072));
}
