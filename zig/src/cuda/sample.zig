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
    pitched_fn: ?Function = null,
    logprob_rows_fn: Function,
    topk_slices_fn: Function,
    topk_merge_fn: Function,
    /// tf_draw_rows; null in a kernel set built before it, where drawRows falls back to one launch a row.
    draw_rows_fn: ?Function = null,
    /// tf_draw_slice_top and tf_draw_rows_finish: a window's rows over slices (null: one block a row).
    slices: ?[2]Function = null,

    /// sample.cu's SW: a slice's entries.
    pub const draw_slice = 8192;
    /// The scratch bytes drawRows' slices take a row: its candidates' keys and ids.
    pub const row_scratch = 2 * 1024 * 4;

    /// tf_topk_slices' slice: a row's entries a block in topkRows' first stage.
    pub const topk_slice = 4096;

    pub fn resolve(mod: Module) !Draws {
        return .{ .draw_fn = try mod.function("tf_draw"), .draw_ids_fn = try mod.function("tf_draw_ids"), .logprob_rows_fn = try mod.function("tf_logprob_rows"), .topk_slices_fn = try mod.function("tf_topk_slices"), .topk_merge_fn = try mod.function("tf_topk_merge"), .draw_rows_fn = mod.function("tf_draw_rows") catch null, .slices = slicesOf(mod) };
    }

    fn slicesOf(mod: Module) ?[2]Function {
        var out: [2]Function = undefined;
        for (&out, [_][:0]const u8{ "tf_draw_slice_top", "tf_draw_rows_finish" }) |*f, name| f.* = mod.function(name) catch return null;
        return out;
    }

    /// Resolve the opt-in pitched entry separately; old images keep the exact legacy symbol lookups.
    pub fn resolveProposal(mod: Module) !Draws {
        var d = try resolve(mod);
        d.pitched_fn = try mod.function("tf_draw_pitched");
        return d;
    }

    /// The bytes topkRows' `parts` holds for `rows` rows of `vocab` at `k`.
    pub fn topkPartsBytes(vocab: usize, k: usize, rows: usize) usize {
        return rows * (std.math.divCeil(usize, vocab, topk_slice) catch unreachable) * k * 8;
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

    /// Per-view logical columns and pitch; draw_ids keeps noise, ties and outputs keyed by global token ID.
    pub fn proposal(d: *const Draws, s: Stream, logits: u64, view: anytype, ids: ?u64, rule_ptr: u64, meta: u64, offset: usize, out: u64, rows: usize, prob: ?u64) !void {
        if (@hasField(@TypeOf(view), "dtype")) if (view.dtype != .bf16) return error.ProposalBasisUnsupported;
        if (view.logical == 0 or view.logical > view.vocab or view.stride < view.logical or (view.ids != null) != (ids != null)) return error.ProposalScoreShape;
        if (view.ids) |map| if (map.len != view.logical) return error.ProposalScoreShape;
        if (view.ids == null and view.logical != view.vocab) return error.ProposalScoreShape;
        if (view.stride == view.logical) return d.draw(s, logits, view.logical, rule_ptr, meta, offset, out, rows, ids, prob);
        const f = d.pitched_fn orelse return error.ProposalServiceUnsupported;
        var a: launch_.Args = .{};
        a.add(logits);
        a.add(@as(u32, @intCast(view.logical)));
        a.add(@as(u32, @intCast(view.stride)));
        a.add(rule_ptr);
        a.add(meta);
        a.add(int(offset));
        a.add(out);
        a.add(ids orelse 0);
        a.add(prob orelse 0);
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
    }

    /// `rows` consecutive bf16 logits rows, row r under the rule at rule_ptr + r and the position at meta + r (plus 1).
    pub fn drawRows(d: *const Draws, s: Stream, logits: u64, vocab: usize, rule_ptr: u64, meta: u64, out: u64, rows: usize, scratch: ?u64) !void {
        if (rows == 0) return;
        if (d.slices) |f| if (scratch) |ck| {
            // each slice's top_k over (slices, rows), then a block a row sorts them and draws
            const ci = ck + rows * 1024 * 4;
            var a: launch_.Args = .{};
            inline for (.{ logits, @as(u32, @intCast(vocab)), rule_ptr, ck, ci }) |v| a.add(v);
            try launch_.launch(f[0], .{ .grid = .{ .x = @intCast((vocab + draw_slice - 1) / draw_slice), .y = @intCast(rows), .z = 1 }, .block = .{ .x = 256 } }, s, &a);
            var b: launch_.Args = .{};
            inline for (.{ logits, @as(u32, @intCast(vocab)), rule_ptr, meta, ck, ci, out }) |v| b.add(v);
            try launch_.launch(f[1], .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 1024 } }, s, &b);
            return;
        };
        const f = d.draw_rows_fn orelse {
            for (0..rows) |i| try d.draw(s, logits + i * vocab * 2, vocab, rule_ptr + i * @sizeOf(Rule), meta + i * 4, 0, out + i * 4, 1, null, null);
            return;
        };
        var a: launch_.Args = .{};
        a.add(logits);
        a.add(@as(u32, @intCast(vocab)));
        a.add(rule_ptr);
        a.add(meta);
        a.add(out);
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

    /// logprobRows' `k` best ids and logits alone (other words zero), many blocks a row; `parts`: topkPartsBytes.
    pub fn topkRows(d: *const Draws, s: Stream, logits: u64, vocab: usize, k: u8, parts: u64, out: u64, rows: usize) !void {
        const slices = std.math.divCeil(usize, vocab, topk_slice) catch unreachable;
        const pv = parts;
        const pi = parts + rows * slices * k * 4;
        var a: launch_.Args = .{};
        a.add(logits);
        a.add(@as(u32, @intCast(vocab)));
        a.add(@as(u32, k));
        a.add(@as(u32, @intCast(slices)));
        a.add(pv);
        a.add(pi);
        try launch_.launch(d.topk_slices_fn, .{ .grid = .{ .x = @intCast(slices), .y = @intCast(rows), .z = 1 }, .block = .{ .x = 256 } }, s, &a);
        var b: launch_.Args = .{};
        b.add(pv);
        b.add(pi);
        b.add(@as(u32, @intCast(slices * k)));
        b.add(@as(u32, k));
        b.add(out);
        try launch_.launch(d.topk_merge_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &b);
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
