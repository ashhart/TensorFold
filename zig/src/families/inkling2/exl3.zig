//! Inkling 2's EXL3 tensors (exllamav3 1.x, mul1, an inkling2 architecture added to it): rank cuts, grouped experts.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const shard = @import("shard.zig");
const metadata = @import("exl3_config.zig");
const experts_run = @import("experts.zig");

const Tensor = core.checkpoint.Tensor;

/// The mul1 codebook marker's value.
const mul1_magic: u32 = 0x83DCD12D;

/// What a rank takes of a projection: all of it, its output columns, or its input rows (each a multiple of 128).
pub const Cut = union(enum) { whole, cols: shard.Span, rows: shard.Span };

/// The device side of a load: uploads go through `put`, dense weights reserve their scratch on `dense`.
pub const Sink = struct {
    ctx: *anyopaque,
    put: *const fn (ctx: *anyopaque, bytes: []const u8) anyerror!u64,
    dense: *cuda.exl3.Dense,
    stream: cuda.Stream,
};

/// config.json's quantization_config: EXL3 from exllamav3 1.x on mul1 (each tensor's width is checked as it loads).
pub fn admit(gpa: std.mem.Allocator, config_json: []const u8) !void {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, config_json, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.NotExl3;
    const q = doc.value.object.get("quantization_config") orelse return error.NotExl3;
    if (q != .object) return error.NotExl3;
    const method = q.object.get("quant_method") orelse return error.NotExl3;
    if (method != .string or !std.mem.eql(u8, method.string, "exl3")) return error.NotExl3;
    try metadata.admit(q.object, false);
}

/// One stored projection's tensors, checked: trellis [k/16][n/16][16 * bits] i16, the mul1 marker, fp16 suh and svh.
pub const Group = struct {
    trellis: Tensor,
    suh: Tensor,
    svh: Tensor,
    halves: u32,

    fn tile(g: Group) usize {
        return g.halves * 16;
    }
};

pub fn group(ck: *core.Checkpoint, base: []const u8, n: usize, k: usize) !Group {
    var buf: [192]u8 = undefined;
    const t = try ck.get(try std.fmt.bufPrint(&buf, "{s}.trellis", .{base}));
    if (t.dtype != .i16 or t.rank != 3 or t.dim(0) != k / 16 or t.dim(1) != n / 16 or t.dim(2) % 8 != 0) {
        std.log.err("inkling2: {s}.trellis is not [{d}, {d}, 16 * bits]", .{ base, k / 16, n / 16 });
        return error.UnexpectedInkling2Tensor;
    }
    const halves = t.dim(2) / 8;
    if (halves < 2 or halves > 16) return error.UnsupportedExl3;
    const marker = try ck.get(try std.fmt.bufPrint(&buf, "{s}.mul1", .{base}));
    if (marker.dtype != .i32 or marker.bytes.len != 4 or std.mem.readInt(u32, marker.bytes[0..4], .little) != mul1_magic) return error.UnsupportedExl3;
    const suh = try ck.get(try std.fmt.bufPrint(&buf, "{s}.suh", .{base}));
    const svh = try ck.get(try std.fmt.bufPrint(&buf, "{s}.svh", .{base}));
    if (!suh.is(.f16, &.{k}) or !svh.is(.f16, &.{n})) return error.UnexpectedInkling2Tensor;
    return .{ .trellis = t, .suh = suh, .svh = svh, .halves = @intCast(halves) };
}

/// The cut's shape (k, n) and, appended to `out`, its trellis words, suh and svh in the stored layout.
pub fn cut(gpa: std.mem.Allocator, g: Group, n: usize, k: usize, c: Cut, out: *std.ArrayList(u8), suh: *std.ArrayList(u8), svh: *std.ArrayList(u8)) ![2]usize {
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
pub fn linear(gpa: std.mem.Allocator, ck: *core.Checkpoint, sink: Sink, base: []const u8, n: usize, k: usize, c: Cut) !cuda.exl3.Weight {
    const g = try group(ck, base, n, k);
    if (cuda.exl3.variantIndex(g.halves, .mul1, 4) == null) return error.UnsupportedExl3;
    var tr: std.ArrayList(u8) = .empty;
    defer tr.deinit(gpa);
    var su: std.ArrayList(u8) = .empty;
    defer su.deinit(gpa);
    var sv: std.ArrayList(u8) = .empty;
    defer sv.deinit(gpa);
    const kn = try cut(gpa, g, n, k, c, &tr, &su, &sv);
    const st = cuda.exl3.strides(kn[0], kn[1], g.halves, false);
    const w: cuda.exl3.Weight = .{
        .trellis = try sink.put(sink.ctx, tr.items),
        .suh = try sink.put(sink.ctx, su.items),
        .svh = try sink.put(sink.ctx, sv.items),
        .k = @intCast(kn[0]),
        .n = @intCast(kn[1]),
        .halves = g.halves,
        .codebook = .mul1,
        .stride_k = st.k,
        .stride_nb = st.nb,
    };
    try sink.dense.reserve(sink.stream, w);
    return w;
}

/// A layer's routed experts on the device and the least and most half-bit widths among their tensors.
pub const Experts = struct { layer: cuda.exl3_experts.Layer, bits: experts_run.Bits };

/// Layer `i`'s routed experts as one grouped EXL3 layer: gate and up by the rank's columns, down by its rows.
pub fn experts(gpa: std.mem.Allocator, ck: *core.Checkpoint, sink: Sink, prefix: []const u8, i: u32, count: usize, d: usize, inter: usize, span: shard.Span) !Experts {
    return expertLayer(gpa, ck, sink, i, count, d, inter, .{ .rank = .{ .prefix = prefix, .span = span } });
}

/// Uploads Mini's whole projections into its text weight arena before dense scratch is reserved.
pub const MiniSink = struct {
    ctx: *anyopaque,
    put: *const fn (ctx: *anyopaque, bytes: []const u8) anyerror!u64,
};

pub fn miniLinear(ck: *core.Checkpoint, sink: MiniSink, widths: std.json.ObjectMap, name: []const u8, n: usize, k: usize) !cuda.exl3.Weight {
    const g = try checkedGroup(ck, widths, name, n, k);
    if (cuda.exl3.variantIndex(g.halves, .mul1, 4) == null) return error.UnsupportedExl3;
    const st = cuda.exl3.strides(k, n, g.halves, false);
    return .{
        .trellis = try sink.put(sink.ctx, g.trellis.bytes),
        .suh = try sink.put(sink.ctx, g.suh.bytes),
        .svh = try sink.put(sink.ctx, g.svh.bytes),
        .k = @intCast(k),
        .n = @intCast(n),
        .halves = g.halves,
        .codebook = .mul1,
        .stride_k = st.k,
        .stride_nb = st.nb,
    };
}

pub fn miniExperts(gpa: std.mem.Allocator, ck: *core.Checkpoint, sink: MiniSink, widths: std.json.ObjectMap, i: u32, count: usize, d: usize, inter: usize) !Experts {
    return expertLayer(gpa, ck, sink, i, count, d, inter, .{ .mini = widths });
}

const ExpertLayout = union(enum) {
    rank: struct { prefix: []const u8, span: shard.Span },
    mini: std.json.ObjectMap,
};

fn expertLayer(gpa: std.mem.Allocator, ck: *core.Checkpoint, sink: anytype, i: u32, count: usize, d: usize, inter: usize, layout: ExpertLayout) !Experts {
    const mods = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };
    var ptrs: [3]u64 = undefined;
    var k2: [3]u64 = undefined;
    var suh: [3]u64 = undefined;
    var svh: [3]u64 = undefined;
    var base_buf: [160]u8 = undefined;
    var bits: experts_run.Bits = .{ .lo = 16, .hi = 0 };
    for (mods, 0..) |mod, m| {
        const down = m == 2;
        const n = if (down) d else if (layout == .mini) 2 * inter else inter;
        const k = if (down) inter else d;
        const c: Cut = switch (layout) {
            .rank => |rank| if (down) .{ .rows = rank.span } else .{ .cols = rank.span },
            .mini => if (down) .whole else .{ .cols = .{ .lo = m * inter, .hi = (m + 1) * inter } },
        };
        var tr: std.ArrayList(u8) = .empty;
        defer tr.deinit(gpa);
        var su: std.ArrayList(u8) = .empty;
        defer su.deinit(gpa);
        var sv: std.ArrayList(u8) = .empty;
        defer sv.deinit(gpa);
        const at = try gpa.alloc(usize, count);
        defer gpa.free(at);
        const widths = try gpa.alloc(i32, count);
        defer gpa.free(widths);
        for (0..count) |e| {
            const base = switch (layout) {
                .rank => |rank| try std.fmt.bufPrint(&base_buf, "{s}layers.{d}.mlp.experts.{d}.{s}", .{ rank.prefix, i, e, mod }),
                .mini => try std.fmt.bufPrint(&base_buf, "{d}.experts.{s}.{d}", .{ i, if (down) mod else "gate_up_proj", e }),
            };
            const g = switch (layout) {
                .rank => try group(ck, base, n, k),
                .mini => |table| try checkedGroup(ck, table, base, n, k),
            };
            at[e] = tr.items.len;
            widths[e] = @intCast(g.halves);
            bits = .{ .lo = @min(bits.lo, g.halves), .hi = @max(bits.hi, g.halves) };
            _ = try cut(gpa, g, n, k, c, &tr, &su, &sv);
        }
        const words = try sink.put(sink.ctx, tr.items);
        const table = try gpa.alloc(u64, count);
        defer gpa.free(table);
        for (table, at) |*t, off| t.* = words + off;
        ptrs[m] = try sink.put(sink.ctx, std.mem.sliceAsBytes(table));
        k2[m] = try sink.put(sink.ctx, std.mem.sliceAsBytes(widths));
        suh[m] = if (layout == .mini and m == 1) suh[0] else try sink.put(sink.ctx, su.items);
        svh[m] = try sink.put(sink.ctx, sv.items);
        if (layout == .rank) ck.release();
    }
    return .{ .bits = bits, .layer = .{
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
        .width = if (layout == .rank) layout.rank.span.len() else inter,
        .codebook = .mul1,
    } };
}

pub fn sidecarHalves(bits: std.json.ObjectMap, base: []const u8) !u32 {
    const value = bits.get(base) orelse {
        std.log.err("Inkling 2 Mini EXL3: missing bit width for {s}", .{base});
        return error.BadExl3Bits;
    };
    const count = 2 * switch (value) {
        .integer => @as(f64, @floatFromInt(value.integer)),
        .float => value.float,
        else => return error.BadExl3Bits,
    };
    if (!std.math.isFinite(count) or count < 2 or count > 16 or count != @trunc(count)) return error.BadExl3Bits;
    return @intFromFloat(count);
}

fn checkedGroup(ck: *core.Checkpoint, widths: std.json.ObjectMap, base: []const u8, n: usize, k: usize) !Group {
    const g = try group(ck, base, n, k);
    if (g.halves != try sidecarHalves(widths, base)) {
        std.log.err("Inkling 2 Mini EXL3: {s} bit width disagrees with its trellis", .{base});
        return error.BadExl3Bits;
    }
    return g;
}

test "exl3: admit takes exllamav3 1.x on mul1 and refuses other codebooks, versions and formats" {
    const a = std.testing.allocator;
    try admit(a, "{\"quantization_config\":{\"quant_method\":\"exl3\",\"version\":\"1.6.6\",\"codebook\":\"mul1\"}}");
    try std.testing.expectError(error.UnsupportedExl3, admit(a, "{\"quantization_config\":{\"quant_method\":\"exl3\",\"version\":\"1.6.6\",\"codebook\":\"mcg\"}}"));
    try std.testing.expectError(error.UnsupportedExl3, admit(a, "{\"quantization_config\":{\"quant_method\":\"exl3\",\"version\":\"0.9\",\"codebook\":\"mul1\"}}"));
    try std.testing.expectError(error.NotExl3, admit(a, "{\"text_config\":{}}"));
    const mini = try std.json.parseFromSlice(std.json.Value, a, "{\"codebook\":\"mul1\"}", .{});
    defer mini.deinit();
    try metadata.admit(mini.value.object, true);
    try std.testing.expectError(error.UnsupportedExl3, metadata.admit(mini.value.object, false));
}

test "exl3: a cut takes a rank's tile columns of every k row, or its whole tile rows, with the matching scales" {
    const a = std.testing.allocator;
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
    try std.testing.expectEqual(@as(u8, 8), tr.items[0]);
    try std.testing.expectEqual(@as(u8, 16 + 8), tr.items[8 * 32]);
    try std.testing.expectEqualSlices(u8, scales[256..], sv.items);
    tr.clearRetainingCapacity();
    su.clearRetainingCapacity();
    sv.clearRetainingCapacity();
    try std.testing.expectEqual([2]usize{ 128, 256 }, try cut(a, g, 256, 256, .{ .rows = .{ .lo = 128, .hi = 256 } }, &tr, &su, &sv));
    try std.testing.expectEqual(@as(u8, 8 * 16), tr.items[0]);
    try std.testing.expectEqualSlices(u8, scales[256..], su.items);
    try std.testing.expectError(error.BadSplit, cut(a, g, 256, 256, .{ .cols = .{ .lo = 64, .hi = 192 } }, &tr, &su, &sv));
}
