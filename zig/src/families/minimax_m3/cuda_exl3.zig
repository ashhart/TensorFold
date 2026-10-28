//! A TP rank's share of a published EXL3 MiniMax-M3.1-Flash (exllamav3 1.x, mul1): projections and routed experts.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");

const Tensor = core.checkpoint.Tensor;
const Loader = weights.Loader;
const prefix = weights.prefix;

/// The mul1 codebook marker's value.
const mul1_magic: u32 = 0x83DCD12D;

/// config.json's quantization_config: EXL3 from exllamav3 1.x on mul1 (each tensor's width is checked as it loads).
pub fn admit(gpa: std.mem.Allocator, config_json: []const u8) !void {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, config_json, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.NotExl3;
    const q = doc.value.object.get("quantization_config") orelse return error.NotExl3;
    if (q != .object) return error.NotExl3;
    const method = q.object.get("quant_method") orelse return error.NotExl3;
    if (method != .string or !std.mem.eql(u8, method.string, "exl3")) return error.NotExl3;
    const book = q.object.get("codebook") orelse return error.UnsupportedExl3;
    if (book != .string or !std.mem.eql(u8, book.string, "mul1")) return error.UnsupportedExl3;
    const version = q.object.get("version") orelse return error.UnsupportedExl3;
    if (version != .string or !std.mem.startsWith(u8, version.string, "1.")) return error.UnsupportedExl3;
}

/// What a rank takes of a projection: all of it, its output columns, or its input rows (each a multiple of 128).
pub const Cut = union(enum) { whole, cols: shard.Span, rows: shard.Span };

/// One stored projection's tensors, checked: trellis [k/16][n/16][16 * bits] i16, the mul1 marker, fp16 suh and svh.
const Group = struct {
    trellis: Tensor,
    suh: Tensor,
    svh: Tensor,
    halves: u32,

    /// A 16x16 tile's bytes: 16 * bits int16 words.
    fn tile(g: Group) usize {
        return g.halves * 16;
    }
};

fn group(L: *Loader, base: []const u8, n: usize, k: usize) !Group {
    const t = try L.get("{s}.trellis", .{base});
    if (t.dtype != .i16 or t.rank != 3 or t.dim(0) != k / 16 or t.dim(1) != n / 16 or t.dim(2) % 8 != 0) return error.UnexpectedMiniMaxTensor;
    const halves = t.dim(2) / 8;
    if (halves < 2 or halves > 16) return error.UnsupportedExl3;
    const marker = try L.get("{s}.mul1", .{base});
    if (marker.dtype != .i32 or marker.bytes.len != 4 or std.mem.readInt(u32, marker.bytes[0..4], .little) != mul1_magic) return error.UnsupportedExl3;
    const suh = try L.get("{s}.suh", .{base});
    const svh = try L.get("{s}.svh", .{base});
    if (!suh.is(.f16, &.{k}) or !svh.is(.f16, &.{n})) return error.UnexpectedMiniMaxTensor;
    return .{ .trellis = t, .suh = suh, .svh = svh, .halves = @intCast(halves) };
}

/// The cut's shape (k, n) and, appended to `out`, its trellis words, suh and svh in the stored layout.
fn cut(gpa: std.mem.Allocator, g: Group, n: usize, k: usize, c: Cut, out: *std.ArrayList(u8), suh: *std.ArrayList(u8), svh: *std.ArrayList(u8)) ![2]usize {
    const tb = g.tile();
    const nt = n / 16;
    const tr = g.trellis.bytes;
    switch (c) {
        .whole => {
            try out.appendSlice(gpa, tr);
            try suh.appendSlice(gpa, g.suh.bytes);
            try svh.appendSlice(gpa, g.svh.bytes);
            return .{ k, n };
        },
        .cols => |s| {
            if (s.lo % 128 != 0 or s.len() % 128 != 0 or s.hi > n or s.len() == 0) return error.BadSplit;
            for (0..k / 16) |kt| try out.appendSlice(gpa, tr[(kt * nt + s.lo / 16) * tb ..][0 .. s.len() / 16 * tb]);
            try suh.appendSlice(gpa, g.suh.bytes);
            try svh.appendSlice(gpa, g.svh.bytes[2 * s.lo .. 2 * s.hi]);
            return .{ k, s.len() };
        },
        .rows => |s| {
            if (s.lo % 128 != 0 or s.len() % 128 != 0 or s.hi > k or s.len() == 0) return error.BadSplit;
            try out.appendSlice(gpa, tr[s.lo / 16 * nt * tb .. s.hi / 16 * nt * tb]);
            try suh.appendSlice(gpa, g.suh.bytes[2 * s.lo .. 2 * s.hi]);
            try svh.appendSlice(gpa, g.svh.bytes);
            return .{ s.len(), n };
        },
    }
}

/// The stored [n][k] projection under `base`, cut to the rank, on the device for cuda.exl3.Dense.
pub fn linear(L: *Loader, base: []const u8, n: usize, k: usize, c: Cut) !cuda.exl3.Weight {
    const g = try group(L, base, n, k);
    if (cuda.exl3.variantIndex(g.halves, .mul1, 4) == null) return error.UnsupportedExl3;
    var tr: std.ArrayList(u8) = .empty;
    defer tr.deinit(L.gpa);
    var su: std.ArrayList(u8) = .empty;
    defer su.deinit(L.gpa);
    var sv: std.ArrayList(u8) = .empty;
    defer sv.deinit(L.gpa);
    const kn = try cut(L.gpa, g, n, k, c, &tr, &su, &sv);
    const st = cuda.exl3.strides(kn[0], kn[1], g.halves, false);
    const w: cuda.exl3.Weight = .{
        .trellis = try L.put(tr.items),
        .suh = try L.put(su.items),
        .svh = try L.put(sv.items),
        .k = @intCast(kn[0]),
        .n = @intCast(kn[1]),
        .halves = g.halves,
        .codebook = .mul1,
        .stride_k = st.k,
        .stride_nb = st.nb,
    };
    try L.t.dense.reserve(L.t.stream, w);
    return w;
}

/// Layer i's routed experts, grouped EXL3: w1 (gate), w3 (up) by the rank's columns, w2 (down) by its rows; SwiGLU-OAI.
pub fn experts(L: *Loader, i: u32) !cuda.experts.Layer {
    const c = &L.w.config;
    const d: usize = c.hidden;
    const inter: usize = c.moe_intermediate;
    const span = L.w.share.moe;
    const count: usize = c.experts;
    const mods = [3][]const u8{ "w1", "w3", "w2" };
    var ptrs: [3]u64 = undefined;
    var k2: [3]u64 = undefined;
    var suh: [3]u64 = undefined;
    var svh: [3]u64 = undefined;
    var base_buf: [128]u8 = undefined;
    for (mods, 0..) |mod, m| {
        const down = m == 2;
        const n = if (down) d else inter;
        const k = if (down) inter else d;
        var tr: std.ArrayList(u8) = .empty;
        defer tr.deinit(L.gpa);
        var su: std.ArrayList(u8) = .empty;
        defer su.deinit(L.gpa);
        var sv: std.ArrayList(u8) = .empty;
        defer sv.deinit(L.gpa);
        const at = try L.gpa.alloc(usize, count);
        defer L.gpa.free(at);
        const widths = try L.gpa.alloc(i32, count);
        defer L.gpa.free(widths);
        for (0..count) |e| {
            const base = try std.fmt.bufPrint(&base_buf, prefix ++ "model.layers.{d}.block_sparse_moe.experts.{d}.{s}", .{ i, e, mod });
            const g = try group(L, base, n, k);
            at[e] = tr.items.len;
            widths[e] = @intCast(g.halves);
            _ = try cut(L.gpa, g, n, k, if (down) .{ .rows = span } else .{ .cols = span }, &tr, &su, &sv);
        }
        const words = try L.put(tr.items);
        const table = try L.gpa.alloc(u64, count);
        defer L.gpa.free(table);
        for (table, at) |*t, off| t.* = words + off;
        ptrs[m] = try L.put(std.mem.sliceAsBytes(table));
        k2[m] = try L.put(std.mem.sliceAsBytes(widths));
        suh[m] = try L.put(su.items);
        svh[m] = try L.put(sv.items);
        L.ck.release();
    }
    const ni = span.len();
    const lay: cuda.exl3_experts.Layer = .{
        .gate = ptrs[0],
        .up = ptrs[1],
        .down = ptrs[2],
        .gate_k2 = k2[0],
        .up_k2 = k2[1],
        .down_k2 = k2[2],
        .suh_g = suh[0],
        .suh_u = suh[1],
        .svh_g = svh[0],
        .svh_u = svh[1],
        .suh_d = suh[2],
        .svh_d = svh[2],
        .experts = count,
        .dims = d,
        .width = ni,
        .codebook = .mul1,
        .limit = c.swiglu_limit,
        .act_mode = 2,
        .alpha = c.swiglu_alpha,
    };
    return .{ .format = .exl3, .exl3 = lay, .up = 0, .down = 0, .width = ni, .dims = d, .experts = count, .limit = c.swiglu_limit, .swiglu = true };
}

/// An fp16 [n][k] matrix's rows [lo, hi) as bf16 into `out`, refused unless exact (the pack's indexer always is).
pub fn f16Rows(gpa: std.mem.Allocator, t: Tensor, lo: usize, hi: usize, out: *std.ArrayList(u16)) !void {
    const k = t.dim(1);
    for (lo * k..hi * k) |j| {
        const h: f16 = @bitCast(std.mem.readInt(u16, t.bytes[2 * j ..][0..2], .little));
        const f: u32 = @bitCast(@as(f32, @floatCast(h)));
        if (f & 0xffff != 0) return error.InexactBf16;
        try out.append(gpa, @intCast(f >> 16));
    }
}

test "admit takes exllamav3 1.x on mul1 and refuses other codebooks, versions and formats" {
    const a = std.testing.allocator;
    try admit(a, "{\"quantization_config\":{\"quant_method\":\"exl3\",\"version\":\"1.6.1\",\"codebook\":\"mul1\"}}");
    try std.testing.expectError(error.UnsupportedExl3, admit(a, "{\"quantization_config\":{\"quant_method\":\"exl3\",\"version\":\"1.6.1\",\"codebook\":\"mcg\"}}"));
    try std.testing.expectError(error.UnsupportedExl3, admit(a, "{\"quantization_config\":{\"quant_method\":\"exl3\",\"version\":\"0.9\",\"codebook\":\"mul1\"}}"));
    try std.testing.expectError(error.NotExl3, admit(a, "{\"text_config\":{}}"));
}

test "a cut takes a rank's tile columns of every k row, or its whole tile rows, with the matching scales" {
    const a = std.testing.allocator;
    // k = 256, n = 256 at 1 bit (2 halves, 32-byte tiles): 16 x 16 tiles, each tile's bytes its (kt, nt) index
    var words: [16 * 16 * 32]u8 = undefined;
    for (0..16) |kt| for (0..16) |nt| @memset(words[(kt * 16 + nt) * 32 ..][0..32], @intCast(kt * 16 + nt));
    var scales: [512]u8 = undefined;
    for (&scales, 0..) |*s, j| s.* = @intCast(j / 2 % 256);
    const t: Tensor = .{ .dtype = .i16, .rank = 3, .shape = .{ 16, 16, 16, 1, 1 }, .bytes = &words };
    const v: Tensor = .{ .dtype = .f16, .rank = 1, .shape = .{ 256, 1, 1, 1, 1 }, .bytes = &scales };
    const g: Group = .{ .trellis = t, .suh = v, .svh = v, .halves = 2 };
    var tr: std.ArrayList(u8) = .empty;
    defer tr.deinit(a);
    var su: std.ArrayList(u8) = .empty;
    defer su.deinit(a);
    var sv: std.ArrayList(u8) = .empty;
    defer sv.deinit(a);
    try std.testing.expectEqual([2]usize{ 256, 128 }, try cut(a, g, 256, 256, .{ .cols = .{ .lo = 128, .hi = 256 } }, &tr, &su, &sv));
    try std.testing.expectEqual(@as(usize, 16 * 8 * 32), tr.items.len);
    try std.testing.expectEqual(@as(u8, 8), tr.items[0]); // k row 0, tile column 8
    try std.testing.expectEqual(@as(u8, 16 + 8), tr.items[8 * 32]); // k row 1 follows
    try std.testing.expectEqual(@as(usize, 512), su.items.len);
    try std.testing.expectEqualSlices(u8, scales[256..], sv.items);
    tr.clearRetainingCapacity();
    su.clearRetainingCapacity();
    sv.clearRetainingCapacity();
    try std.testing.expectEqual([2]usize{ 128, 256 }, try cut(a, g, 256, 256, .{ .rows = .{ .lo = 128, .hi = 256 } }, &tr, &su, &sv));
    try std.testing.expectEqual(@as(u8, 8 * 16), tr.items[0]); // tile row 8 first
    try std.testing.expectEqualSlices(u8, scales[256..], su.items);
    try std.testing.expectError(error.BadSplit, cut(a, g, 256, 256, .{ .cols = .{ .lo = 64, .hi = 192 } }, &tr, &su, &sv));
}
