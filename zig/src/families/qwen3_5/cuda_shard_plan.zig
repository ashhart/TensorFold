//! Which output columns of each dense Qwen3.5 EXL3 weight a rank cuts, and the K splits its launches keep.

const std = @import("std");
const cuda = @import("cuda");
const cs = @import("cuda_shard.zig");

const exl3 = cuda.exl3;

/// The EXL3 weights of the dense hybrids and the head: each cut along its n (outputs), every rank reading the whole k.
pub const Part = enum { q_proj, k_proj, v_proj, o_proj, in_proj_qkv, in_proj_z, out_proj, gate_proj, up_proj, down_proj, lm_head };

/// A whole weight's shape: n outputs of k inputs.
pub const Dims = struct { n: usize, k: usize };

/// `part`'s whole shape at the widths of `s`.
pub fn dims(s: cs.Shape, part: Part) Dims {
    return switch (part) {
        .q_proj => .{ .n = s.q_heads * 2 * cs.head_dim, .k = s.hidden },
        .k_proj, .v_proj => .{ .n = s.kv_heads * cs.head_dim, .k = s.hidden },
        .o_proj => .{ .n = s.hidden, .k = s.q_heads * cs.head_dim },
        .in_proj_qkv => .{ .n = (2 * s.k_heads + s.v_heads) * cs.dk, .k = s.hidden },
        .in_proj_z => .{ .n = s.v_heads * cs.dk, .k = s.hidden },
        .out_proj => .{ .n = s.hidden, .k = s.v_heads * cs.dk },
        .gate_proj, .up_proj => .{ .n = s.intermediate, .k = s.hidden },
        .down_proj => .{ .n = s.hidden, .k = s.intermediate },
        .lm_head => .{ .n = s.vocab, .k = s.hidden },
    };
}

/// A weight of `d`'s shape with no storage behind it: all that `planOf` and the column views read.
fn unplaced(d: Dims) exl3.Weight {
    return .{ .trellis = 0, .suh = 0, .svh = 0, .k = @intCast(d.k), .n = @intCast(d.n), .halves = 8, .codebook = .mul1, .stride_k = 0, .stride_nb = 0 };
}

/// One weight as a rank cuts it: its spans of the whole n, and the whole weight's (sk, wk) for every launch of them.
pub const Cut = struct {
    whole: Dims,
    spans: [3]cs.Span,
    len: usize, // the spans in use: in_proj_qkv's three segments, else one
    sk: u32,
    wk: u32,

    pub fn items(c: *const Cut) []const cs.Span {
        return c.spans[0..c.len];
    }
};

/// `part` as the rank of `share` cuts it; the plan is the whole weight's, the one TP=1 launches.
pub fn cutOf(part: Part, s: cs.Shape, share: cs.Share) Cut {
    const d = dims(s, part);
    const p = exl3.planOf(unplaced(d));
    var spans: [3]cs.Span = @splat(.{ .lo = 0, .hi = 0 });
    var len: usize = 1;
    switch (part) {
        .q_proj => spans[0] = share.q,
        .k_proj, .v_proj => spans[0] = share.kv,
        .o_proj, .out_proj, .down_proj => spans[0] = share.hidden,
        .in_proj_qkv => {
            spans = share.qkv;
            len = 3;
        },
        .in_proj_z => spans[0] = share.z,
        .gate_proj, .up_proj => spans[0] = share.mlp,
        .lm_head => spans[0] = share.vocab,
    }
    return .{ .whole = d, .spans = spans, .len = len, .sk = p.sk, .wk = p.wk };
}

/// A rank's view of `whole` (the whole weight on the device) over `span` of its n, launching the whole's plan.
pub fn view(whole: exl3.Weight, span: cs.Span) !exl3.Weight {
    return whole.pinnedColumns(@intCast(span.lo), @intCast(span.len()));
}

const Plan = struct { sk: u32, wk: u32 };

/// What `columns` would launch for `span`: the plan of the span's own shape, as a weight cut without the pin plans.
fn ownPlan(d: Dims, span: cs.Span) !Plan {
    const v = try unplaced(d).columns(@intCast(span.lo), @intCast(span.len()));
    const p = exl3.planOf(v);
    return .{ .sk = p.sk, .wk = p.wk };
}

fn sp(lo: usize, hi: usize) cs.Span {
    return .{ .lo = lo, .hi = hi };
}

const Want = struct { part: Part, n: usize, k: usize, sk: u32, wk: u32, r0: []const cs.Span, r1: []const cs.Span };

/// The 27B at TP=2, the plan of each whole weight as `exl3.plan` gives it (K splits, warps).
const tp2 = [_]Want{
    .{ .part = .q_proj, .n = 12288, .k = 5120, .sk = 1, .wk = 8, .r0 = &.{sp(0, 6144)}, .r1 = &.{sp(6144, 12288)} },
    .{ .part = .k_proj, .n = 1024, .k = 5120, .sk = 8, .wk = 4, .r0 = &.{sp(0, 512)}, .r1 = &.{sp(512, 1024)} },
    .{ .part = .v_proj, .n = 1024, .k = 5120, .sk = 8, .wk = 4, .r0 = &.{sp(0, 512)}, .r1 = &.{sp(512, 1024)} },
    .{ .part = .o_proj, .n = 5120, .k = 6144, .sk = 8, .wk = 4, .r0 = &.{sp(0, 2560)}, .r1 = &.{sp(2560, 5120)} },
    .{ .part = .in_proj_qkv, .n = 10240, .k = 5120, .sk = 1, .wk = 8, .r0 = &.{ sp(0, 1024), sp(2048, 3072), sp(4096, 7168) }, .r1 = &.{ sp(1024, 2048), sp(3072, 4096), sp(7168, 10240) } },
    .{ .part = .in_proj_z, .n = 6144, .k = 5120, .sk = 4, .wk = 4, .r0 = &.{sp(0, 3072)}, .r1 = &.{sp(3072, 6144)} },
    .{ .part = .out_proj, .n = 5120, .k = 6144, .sk = 8, .wk = 4, .r0 = &.{sp(0, 2560)}, .r1 = &.{sp(2560, 5120)} },
    .{ .part = .gate_proj, .n = 17408, .k = 5120, .sk = 1, .wk = 8, .r0 = &.{sp(0, 8704)}, .r1 = &.{sp(8704, 17408)} },
    .{ .part = .up_proj, .n = 17408, .k = 5120, .sk = 1, .wk = 8, .r0 = &.{sp(0, 8704)}, .r1 = &.{sp(8704, 17408)} },
    .{ .part = .down_proj, .n = 5120, .k = 17408, .sk = 8, .wk = 4, .r0 = &.{sp(0, 2560)}, .r1 = &.{sp(2560, 5120)} },
    .{ .part = .lm_head, .n = 248320, .k = 5120, .sk = 1, .wk = 8, .r0 = &.{sp(0, 124160)}, .r1 = &.{sp(124160, 248320)} },
};

test "TP=2 on the 27B: each weight's spans of n, and the whole weight's (sk, wk) for both halves" {
    for (tp2) |w| for ([_][]const cs.Span{ w.r0, w.r1 }, 0..) |want, rank| {
        const c = cutOf(w.part, cs.qwen27, try cs.Share.of(cs.qwen27, rank, 2));
        try std.testing.expectEqual(Dims{ .n = w.n, .k = w.k }, c.whole);
        try std.testing.expectEqual(Plan{ .sk = w.sk, .wk = w.wk }, Plan{ .sk = c.sk, .wk = c.wk });
        try std.testing.expectEqualSlices(cs.Span, want, c.items());
    };
}

test "at worlds 1, 2 and 4 the spans tile each weight's n in 128-tiles and every cut keeps the whole weight's plan" {
    const keys = cs.qwen27.k_heads * cs.dk; // in_proj_qkv's q and k segments are this wide, v the rest
    for ([_]usize{ 1, 2, 4 }) |world| for (std.enums.values(Part)) |part| {
        const d = dims(cs.qwen27, part);
        const whole = exl3.planOf(unplaced(d));
        const segments: usize = if (part == .in_proj_qkv) 3 else 1;
        const starts = [3]usize{ 0, keys, 2 * keys };
        const ends = [3]usize{ if (segments == 3) keys else d.n, 2 * keys, d.n };
        var next = [3]usize{ 0, 0, 0 };
        for (0..world) |rank| {
            const c = cutOf(part, cs.qwen27, try cs.Share.of(cs.qwen27, rank, world));
            try std.testing.expectEqual(Plan{ .sk = whole.sk, .wk = whole.wk }, Plan{ .sk = c.sk, .wk = c.wk });
            for (c.items(), 0..) |span, j| {
                if (rank == 0) next[j] = starts[j];
                try std.testing.expectEqual(next[j], span.lo);
                try std.testing.expect(span.lo % cs.unit == 0 and span.len() % cs.unit == 0 and span.len() > 0);
                next[j] = span.hi;
            }
        }
        for (0..segments) |j| try std.testing.expectEqual(ends[j], next[j]);
    };
}

/// The weights whose halves (world 2) or quarters (world 4) would plan their own (sk, wk), not the whole's.
const own_at_2 = [_]Part{ .q_proj, .in_proj_qkv, .in_proj_z, .down_proj };
const own_at_4 = [_]Part{ .q_proj, .in_proj_qkv, .in_proj_z, .gate_proj, .up_proj, .down_proj };

test "a half cut without the pin plans its own (sk, wk) for q_proj, in_proj_qkv, in_proj_z and down_proj; the view keeps the whole's" {
    for ([_]usize{ 2, 4 }) |world| for (std.enums.values(Part)) |part| {
        const own_set: []const Part = if (world == 2) &own_at_2 else &own_at_4;
        const listed = std.mem.indexOfScalar(Part, own_set, part) != null;
        for (0..world) |rank| {
            const c = cutOf(part, cs.qwen27, try cs.Share.of(cs.qwen27, rank, world));
            for (c.items()) |span| {
                const pinned = exl3.planOf(try view(unplaced(c.whole), span));
                try std.testing.expectEqual(Plan{ .sk = c.sk, .wk = c.wk }, Plan{ .sk = pinned.sk, .wk = pinned.wk });
                const own = try ownPlan(c.whole, span);
                try std.testing.expectEqual(listed, own.sk != c.sk or own.wk != c.wk);
            }
        }
    };
    // q_proj's half plans 4 splits of 4 warps (the whole 1 of 8), in_proj_z's 8 of 4, down_proj's 16 of 4
    try std.testing.expectEqual(Plan{ .sk = 4, .wk = 4 }, try ownPlan(dims(cs.qwen27, .q_proj), sp(0, 6144)));
    try std.testing.expectEqual(Plan{ .sk = 8, .wk = 4 }, try ownPlan(dims(cs.qwen27, .in_proj_z), sp(0, 3072)));
    try std.testing.expectEqual(Plan{ .sk = 16, .wk = 4 }, try ownPlan(dims(cs.qwen27, .down_proj), sp(0, 2560)));
}
