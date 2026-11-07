//! A TP rank's share of Ling's tensors: EXL3 by whole 128-wide blocks (output columns, input rows), bf16 by rows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const proj = @import("weights_proj.zig");

const Reader = proj.Reader;
const Proj = proj.Proj;
const Tensor = core.checkpoint.Tensor;

/// [lo, hi) of an axis.
pub const Span = struct {
    lo: usize,
    hi: usize,

    pub fn len(s: Span) usize {
        return s.hi - s.lo;
    }
};

/// A linear [n, k]'s part: output rows in the `out` spans, in order (none: every row), input columns `in` (null: all).
pub const Cut = struct { out: []const Span = &.{}, in: ?Span = null };

/// Rank `rank`'s even share of `total`.
pub fn even(total: usize, rank: usize, world: usize) Span {
    const w = total / world;
    return .{ .lo = rank * w, .hi = (rank + 1) * w };
}

/// A rank's config: heads and widths divided by the world, each share whole 128-wide blocks (EXL3's tiles).
pub fn rankConfig(c: config.Config, world: u32, why: *config.Why) !config.Config {
    if (world == 1) return c;
    if (c.quant != .exl3) {
        why.set("TP over {d} ranks reads EXL3 packs only (their 128-wide blocks are the cuts)", .{world});
        return error.TpFormat;
    }
    var r = c;
    r.kda_heads = c.kda_heads / world;
    r.mla_heads = c.mla_heads / world;
    r.dense_inter = c.dense_inter / world;
    r.moe_inter = c.moe_inter / world;
    r.shared_inter = c.shared_inter / world;
    const parts = [_]struct { []const u8, u32, u32, u32 }{
        .{ "KDA heads x 128", c.kda_heads * c.kda_dim, r.kda_heads * c.kda_dim, 1 },
        .{ "MLA heads x qk", c.mla_heads * c.qkDim(), r.mla_heads * c.qkDim(), 1 },
        .{ "intermediate_size", c.dense_inter, r.dense_inter, @intFromBool(c.dense_layers > 0) },
        .{ "moe_intermediate_size", c.moe_inter, r.moe_inter, 1 },
        .{ "the shared expert's width", c.shared_inter, r.shared_inter, @intFromBool(c.shared_inter > 0) },
        .{ "vocab_size", c.vocab, c.vocab / world, 1 },
    };
    for (parts) |p| if (p[3] == 1 and (p[1] % world != 0 or p[2] * world != p[1] or p[2] % 128 != 0 or p[2] == 0)) {
        why.set("{s} {d} does not split into {d} shares of whole 128-wide blocks", .{ p[0], p[1], world });
        return error.TpShape;
    };
    if (c.kda_heads % world != 0 or c.mla_heads % world != 0) {
        why.set("{d} KDA and {d} MLA heads do not split over {d} ranks", .{ c.kda_heads, c.mla_heads, world });
        return error.TpShape;
    }
    return r;
}

fn blocks(s: Span) !void {
    if (s.lo % 128 != 0 or s.hi % 128 != 0 or s.hi <= s.lo) return error.TpCut;
}

/// `t` [rows][row bytes] cut to rows `s` (and the shape to match).
fn rowsOf(t: Tensor, s: Span) Tensor {
    var u = t;
    const row = t.bytes.len / t.shape[0];
    u.bytes = t.bytes[s.lo * row .. s.hi * row];
    u.shape[0] = s.len();
    return u;
}

/// Linear `module` [n, k], cut; whole when the cut is empty (the TP=1 path, as Reader.linear reads it).
pub fn linear(r: *Reader, module: []const u8, n: usize, k: usize, cut: Cut) !Proj {
    if (cut.out.len == 0 and cut.in == null) return r.linear(module, n, k);
    var b: [256]u8 = undefined;
    const trellis = try std.fmt.bufPrint(&b, "{s}.trellis", .{module});
    if (r.c.quant == .exl3 and r.ck.has(trellis)) return .{ .exl3 = try exl3(r, module, n, k, cut) };
    if (r.c.format(module) != .bf16 and r.c.quant != .exl3) return error.TpFormat;
    return plain(r, module, n, k, cut);
}

/// An EXL3 linear's cut as its own stored matrix: each tile row's spans of tile columns, or whole tile rows.
fn exl3(r: *Reader, module: []const u8, n: usize, k: usize, cut: Cut) !cuda.exl3.Weight {
    const g = try r.exl3Group(module, n, k);
    const tile = g.trellis.shape[2] * 2;
    const nt = g.n / 16;
    var host: std.ArrayList(u8) = .empty;
    defer host.deinit(r.gpa);
    var sv: std.ArrayList(u8) = .empty;
    defer sv.deinit(r.gpa);
    var su: []const u8 = g.suh.bytes;
    var kk = k;
    var nn = g.n;
    if (cut.in) |s| {
        if (cut.out.len != 0) return error.TpCut;
        try blocks(s);
        try host.appendSlice(r.gpa, g.trellis.bytes[(s.lo / 16) * nt * tile .. (s.hi / 16) * nt * tile]);
        su = g.suh.bytes[s.lo * 2 .. s.hi * 2];
        try sv.appendSlice(r.gpa, g.svh.bytes);
        kk = s.len();
    } else {
        nn = 0;
        for (cut.out) |s| {
            try blocks(s);
            nn += s.len();
            try sv.appendSlice(r.gpa, g.svh.bytes[s.lo * 2 .. s.hi * 2]);
        }
        for (0..k / 16) |row| for (cut.out) |s| {
            try host.appendSlice(r.gpa, g.trellis.bytes[row * nt * tile + (s.lo / 16) * tile ..][0 .. (s.len() / 16) * tile]);
        };
    }
    const st = cuda.exl3.strides(kk, nn, g.halves, false);
    var nb: [256]u8 = undefined;
    return .{
        .trellis = try r.store.keep(host.items, module),
        .suh = try r.store.keep(su, try std.fmt.bufPrint(&nb, "{s}.suh", .{module})),
        .svh = try r.store.keep(sv.items, try std.fmt.bufPrint(&nb, "{s}.svh", .{module})),
        .k = @intCast(kk),
        .n = @intCast(nn),
        .halves = g.halves,
        .codebook = g.codebook,
        .stride_k = st.k,
        .stride_nb = st.nb,
    };
}

/// A bf16 (or an EXL3 pack's fp16) linear cut by output rows, or by input columns, rounded to bf16 for the lanes.
fn plain(r: *Reader, module: []const u8, n: usize, k: usize, cut: Cut) !Proj {
    var b: [256]u8 = undefined;
    const t = try r.ck.get(try std.fmt.bufPrint(&b, "{s}.weight", .{module}));
    if (!(t.is(.bf16, &.{ n, k }) or t.is(.f16, &.{ n, k }))) return error.UnexpectedTensor;
    var host: std.ArrayList(u8) = .empty;
    defer host.deinit(r.gpa);
    var rows: usize = 0;
    var cols = k;
    if (cut.in) |s| {
        cols = s.len();
        rows = n;
        for (0..n) |i| {
            var u = rowsOf(t, .{ .lo = i, .hi = i + 1 });
            u.bytes = u.bytes[s.lo * 2 .. s.hi * 2];
            u.shape[1] = cols;
            try proj.append(r.gpa, &host, u, .bf16);
        }
    } else for (cut.out) |s| {
        rows += s.len();
        try proj.append(r.gpa, &host, rowsOf(t, s), .bf16);
    }
    var u = t;
    u.dtype = .bf16;
    u.bytes = host.items;
    u.shape[0] = rows;
    u.shape[1] = cols;
    return r.bf16Of(module, u, rows, cols);
}

/// A vector `tensor` of `numel` values (A_log, dt_bias) cut to `s`, as `to`.
pub fn small(r: *Reader, tensor: []const u8, numel: usize, to: core.checkpoint.DType, s: Span) !u64 {
    const t = try r.ck.get(tensor);
    if (t.numel() != numel) return error.UnexpectedTensor;
    var u = t;
    u.rank = 1;
    u.shape[0] = numel;
    var host: std.ArrayList(u8) = .empty;
    defer host.deinit(r.gpa);
    try proj.append(r.gpa, &host, rowsOf(u, s), to);
    return r.store.keep(host.items, tensor);
}

/// The fused q|k|v conv1d [3n][taps] cut to `spans` of channels, fp32.
pub fn convs(r: *Reader, tensor: []const u8, n3: usize, taps: usize, spans: []const Span) !u64 {
    const t = try r.ck.get(tensor);
    if (t.numel() != n3 * taps) return error.UnexpectedTensor;
    var u = t;
    u.rank = 2;
    u.shape[0] = n3;
    u.shape[1] = taps;
    var host: std.ArrayList(u8) = .empty;
    defer host.deinit(r.gpa);
    for (spans) |s| try proj.append(r.gpa, &host, rowsOf(u, s), .f32);
    return r.store.keep(host.items, tensor);
}

/// One expert's trellis cut by width: gate and up by tile columns, down by tile rows (into `a`, or a view).
pub fn expertTrellis(a: std.mem.Allocator, g: Reader.Group, k: usize, by_cols: bool, s: Span) ![]const u8 {
    try blocks(s);
    const tile = g.trellis.shape[2] * 2;
    const nt = g.n / 16;
    if (!by_cols) return g.trellis.bytes[(s.lo / 16) * nt * tile .. (s.hi / 16) * nt * tile];
    const w = (s.len() / 16) * tile;
    const out = try a.alloc(u8, (k / 16) * w);
    for (0..k / 16) |row| @memcpy(out[row * w ..][0..w], g.trellis.bytes[row * nt * tile + (s.lo / 16) * tile ..][0..w]);
    return out;
}

test "a TP=2 config halves heads and widths on whole 128-wide blocks; other worlds and shapes are refused" {
    var why: config.Why = .{};
    var c = try config.parse(std.testing.allocator, @embedFile("fixtures/config.json"), &why);
    try std.testing.expectError(error.TpFormat, rankConfig(c, 2, &why));
    c.quant = .exl3;
    const r = try rankConfig(c, 2, &why);
    try std.testing.expectEqual(@as(u32, 32), r.kda_heads);
    try std.testing.expectEqual(@as(u32, 32), r.mla_heads);
    try std.testing.expectEqual(@as(u32, 640), r.moe_inter);
    try std.testing.expectEqual(@as(u32, 5120), r.dense_inter);
    try std.testing.expectEqual(c.vocab, r.vocab); // the head's half is the loader's; logits stay whole
    try std.testing.expectEqual(Span{ .lo = 78592, .hi = 157184 }, even(c.vocab, 1, 2));
    c.moe_inter = 128;
    try std.testing.expectError(error.TpShape, rankConfig(c, 2, &why));
}
