//! Exact widening of MLX affine projections: codes re-packed at more bits, group tables repeated for smaller groups.

const std = @import("std");
const mlx = @import("mlx.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Host = mlx.Host;
const Width = mlx.Width;
const Tensor = types.Tensor;

pub const Error = error{
    /// The target has fewer bits than the source: codes would not fit.
    NarrowsBits,
    /// The target group does not divide the source group: one scale could not cover a target group.
    GroupDoesNotDivide,
    /// The target is not an affine width the kernels run (bits 2, 3, 4, 5, 6, 8; groups 32, 64, 128).
    UnsupportedQuantization,
    /// The words and tables disagree on shape, or fused members on K or table type.
    UnexpectedTensor,
} || Allocator.Error;

/// The width a set of members widens to exactly: the most bits and the smallest group among them.
pub fn common(widths: []const Width) Width {
    std.debug.assert(widths.len > 0);
    var w = widths[0];
    for (widths[1..]) |x| w = .{ .bits = @max(w.bits, x.bits), .group = @min(w.group, x.group) };
    return w;
}

fn supported(w: Width) bool {
    const bits_ok = std.mem.indexOfScalar(u8, &mlx.bit_widths, w.bits) != null;
    return bits_ok and (w.group == 32 or w.group == 64 or w.group == 128);
}

/// Rows of a host with its leading dimensions folded (a stack's experts times N), after checking the three agree.
fn foldedRows(h: Host) error{UnexpectedTensor}!usize {
    const r = h.words.rank;
    if (r < 2 or h.scales.rank != r or h.biases.rank != r) return error.UnexpectedTensor;
    if (h.scales.dtype != h.biases.dtype) return error.UnexpectedTensor;
    var n: usize = 1;
    for (0..r - 1) |i| {
        if (h.scales.shape[i] != h.words.shape[i] or h.biases.shape[i] != h.words.shape[i]) return error.UnexpectedTensor;
        n *= h.words.shape[i];
    }
    if (h.scales.shape[r - 1] != h.biases.shape[r - 1]) return error.UnexpectedTensor;
    if (h.words.shape[r - 1] * 32 != h.scales.shape[r - 1] * h.group * h.bits) return error.UnexpectedTensor;
    return n;
}

/// The same dequantized values stored at width `to`: never fewer bits, and a group that divides the source's.
pub fn widen(a: Allocator, h: Host, to: Width) Error!Host {
    if (!supported(to)) return error.UnsupportedQuantization;
    const n = try foldedRows(h);
    if (to.bits == h.bits and to.group == h.group) return h;
    if (to.bits < h.bits) return error.NarrowsBits;
    if (h.group % to.group != 0) return error.GroupDoesNotDivide;
    const reps = h.group / to.group;
    return .{
        .words = if (to.bits == h.bits) h.words else try repack(a, h.words, n, h.bits, to.bits),
        .scales = if (reps == 1) h.scales else try repeat(a, h.scales, n, reps),
        .biases = if (reps == 1) h.biases else try repeat(a, h.biases, n, reps),
        .bits = to.bits,
        .group = to.group,
    };
}

/// Words of `bits`-wide codes re-packed at `wider` bits, code for code, low code first (`n` folded rows).
fn repack(a: Allocator, words: Tensor, n: usize, bits: u8, wider: u8) Allocator.Error!Tensor {
    const r = words.rank;
    const from_row = words.shape[r - 1];
    const k = from_row * 32 / bits;
    const to_row = k * wider / 32;
    const out = try a.alloc(u8, n * to_row * 4);
    @memset(out, 0);
    const mask: u64 = (@as(u64, 1) << @intCast(bits)) - 1;
    for (0..n) |row| for (0..k) |c| {
        const code = codeAt(words.bytes, row * from_row, c * bits) & mask;
        const bit = c * wider;
        const at = row * to_row + (bit >> 5);
        const v = code << @intCast(bit & 31);
        orWord(out, at, @truncate(v));
        if ((bit & 31) + wider > 32) orWord(out, at + 1, @truncate(v >> 32));
    };
    var t = words;
    t.shape[r - 1] = to_row;
    t.bytes = out;
    return t;
}

/// The (unmasked) code starting `bit` bits into the row of words beginning at word `base`.
fn codeAt(bytes: []const u8, base: usize, bit: usize) u64 {
    const at = base + (bit >> 5);
    const lo: u64 = std.mem.readInt(u32, bytes[4 * at ..][0..4], .little);
    const hi: u64 = if ((bit & 31) != 0 and 4 * (at + 2) <= bytes.len) std.mem.readInt(u32, bytes[4 * (at + 1) ..][0..4], .little) else 0;
    return (lo | hi << 32) >> @intCast(bit & 31);
}

fn orWord(bytes: []u8, at: usize, v: u32) void {
    const w = bytes[4 * at ..][0..4];
    std.mem.writeInt(u32, w, std.mem.readInt(u32, w, .little) | v, .little);
}

/// A group table with every entry repeated `reps` times along its last dimension (`n` folded rows).
fn repeat(a: Allocator, t: Tensor, n: usize, reps: usize) Allocator.Error!Tensor {
    const size = t.dtype.size();
    const groups = t.shape[t.rank - 1];
    const out = try a.alloc(u8, n * groups * reps * size);
    for (0..n * groups) |g| {
        const src = t.bytes[g * size ..][0..size];
        for (0..reps) |j| @memcpy(out[(g * reps + j) * size ..][0..size], src);
    }
    var o = t;
    o.shape[t.rank - 1] = groups * reps;
    o.bytes = out;
    return o;
}

/// One rank-2 projection of `hosts`' rows in order, all widened to their `common` width; K and tables must agree.
pub fn fuse(a: Allocator, hosts: []const Host) Error!Host {
    if (hosts.len == 0) return error.UnexpectedTensor;
    var to: Width = .{ .bits = hosts[0].bits, .group = hosts[0].group };
    for (hosts[1..]) |h| to = common(&.{ to, .{ .bits = h.bits, .group = h.group } });
    const parts = try a.alloc(Host, hosts.len);
    var n: usize = 0;
    for (hosts, parts) |h, *p| {
        if (h.words.rank != 2 or h.scales.dtype != hosts[0].scales.dtype) return error.UnexpectedTensor;
        if (h.scales.shape[1] * h.group != hosts[0].scales.shape[1] * hosts[0].group) return error.UnexpectedTensor;
        p.* = try widen(a, h, to);
        n += p.words.shape[0];
    }
    if (hosts.len == 1) return parts[0];
    var out = parts[0];
    out.words = try concatRows(a, parts, .words, n);
    out.scales = try concatRows(a, parts, .scales, n);
    out.biases = try concatRows(a, parts, .biases, n);
    return out;
}

fn concatRows(a: Allocator, parts: []const Host, comptime field: std.meta.FieldEnum(Host), n: usize) Allocator.Error!Tensor {
    var total: usize = 0;
    for (parts) |p| total += @field(p, @tagName(field)).bytes.len;
    const out = try a.alloc(u8, total);
    var at: usize = 0;
    for (parts) |p| {
        const b = @field(p, @tagName(field)).bytes;
        @memcpy(out[at..][0..b.len], b);
        at += b.len;
    }
    var t = @field(parts[0], @tagName(field));
    t.shape[0] = n;
    t.bytes = out;
    return t;
}

// ---- tests ----

const testing = std.testing;
const groups_all = [_]u16{ 32, 64, 128 };

fn tensor(dtype: types.DType, shape: []const usize, bytes: []const u8) Tensor {
    var t: Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = bytes };
    @memcpy(t.shape[0..shape.len], shape);
    return t;
}

/// A host of `lead` x `k` codes of width `w` with bf16 tables: row 0 all zeros, row 1 all max, the rest random.
fn sample(a: Allocator, rng: std.Random, lead: []const usize, k: usize, w: Width) !Host {
    var n: usize = 1;
    for (lead) |d| n *= d;
    const row = k * w.bits / 32;
    const words = try a.alloc(u8, n * row * 4);
    @memset(words, 0);
    const max: u64 = (@as(u64, 1) << @intCast(w.bits)) - 1;
    for (0..n) |r| for (0..k) |c| {
        const code: u64 = switch (r % 4) {
            0 => 0,
            1 => max,
            else => rng.int(u64) & max,
        };
        const bit = c * w.bits;
        const v = code << @intCast(bit & 31);
        orWord(words, r * row + (bit >> 5), @truncate(v));
        if ((bit & 31) + w.bits > 32) orWord(words, r * row + (bit >> 5) + 1, @truncate(v >> 32));
    };
    const g = k / w.group;
    const scales = try a.alloc(u8, n * g * 2);
    const biases = try a.alloc(u8, n * g * 2);
    for (0..n * g) |i| {
        std.mem.writeInt(u16, scales[2 * i ..][0..2], bf16Of(rng.floatNorm(f32) * 0.02), .little);
        std.mem.writeInt(u16, biases[2 * i ..][0..2], bf16Of(rng.floatNorm(f32) * 0.1), .little);
    }
    var ws: [5]usize = undefined;
    var ts: [5]usize = undefined;
    @memcpy(ws[0..lead.len], lead);
    @memcpy(ts[0..lead.len], lead);
    ws[lead.len] = row;
    ts[lead.len] = g;
    const r = lead.len + 1;
    return .{
        .words = tensor(.i32, ws[0..r], words),
        .scales = tensor(.bf16, ts[0..r], scales),
        .biases = tensor(.bf16, ts[0..r], biases),
        .bits = w.bits,
        .group = w.group,
    };
}

fn bf16Of(x: f32) u16 {
    return @truncate(@as(u32, @bitCast(x)) >> 16);
}

fn dequantized(a: Allocator, h: Host) ![]f64 {
    const out = try a.alloc(f64, mlx.elements(h));
    try mlx.reference(h, out);
    return out;
}

fn expectBits(want: []const f64, got: []const f64) !void {
    try testing.expectEqual(want.len, got.len);
    for (want, got) |x, y| try testing.expectEqual(@as(u64, @bitCast(x)), @as(u64, @bitCast(y)));
}

test "every width widens to every wider one with the same fp64 values, bit for bit" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x51de);
    const rng = prng.random();
    var checked: usize = 0;
    for (mlx.bit_widths) |bits| for (groups_all) |group| {
        const from: Width = .{ .bits = bits, .group = group };
        const h = try sample(a, rng, &.{6}, 256, from);
        const want = try dequantized(a, h);
        for (mlx.bit_widths) |to_bits| for (groups_all) |to_group| {
            if (to_bits < bits or group % to_group != 0) continue;
            const to: Width = .{ .bits = to_bits, .group = to_group };
            const w = try widen(a, h, to);
            try testing.expectEqual(to_bits, w.bits);
            try testing.expectEqual(to_group, w.group);
            try testing.expectEqual(@as(usize, 256) * to_bits / 32, w.words.shape[1]);
            try testing.expectEqual(@as(usize, 256) / to_group, w.scales.shape[1]);
            try expectBits(want, try dequantized(a, w));
            checked += 1;
        };
    };
    // 18 sources; each reaches the wider-or-equal bits times the groups dividing its own
    try testing.expectEqual(@as(usize, 126), checked);
}

test "a stack of experts widens row for row" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0xe0e1);
    const rng = prng.random();
    const h = try sample(a, rng, &.{ 3, 4 }, 384, .{ .bits = 3, .group = 128 });
    const w = try widen(a, h, .{ .bits = 5, .group = 32 });
    try testing.expectEqual(@as(u8, 3), w.words.rank);
    try testing.expectEqualSlices(usize, &.{ 3, 4, 384 * 5 / 32 }, w.words.shape[0..3]);
    try testing.expectEqualSlices(usize, &.{ 3, 4, 12 }, w.biases.shape[0..3]);
    try expectBits(try dequantized(a, h), try dequantized(a, w));
}

test "the same width is returned as it is" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(1);
    const h = try sample(a, prng.random(), &.{2}, 128, .{ .bits = 4, .group = 64 });
    const w = try widen(a, h, .{ .bits = 4, .group = 64 });
    try testing.expectEqual(h.words.bytes.ptr, w.words.bytes.ptr);
    try testing.expectEqual(h.scales.bytes.ptr, w.scales.bytes.ptr);
}

test "narrowing bits, growing groups and widths the kernels do not run are refused" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(2);
    const rng = prng.random();
    const h = try sample(a, rng, &.{2}, 256, .{ .bits = 5, .group = 64 });
    try testing.expectError(error.NarrowsBits, widen(a, h, .{ .bits = 4, .group = 64 }));
    try testing.expectError(error.NarrowsBits, widen(a, h, .{ .bits = 2, .group = 32 }));
    try testing.expectError(error.GroupDoesNotDivide, widen(a, h, .{ .bits = 5, .group = 128 }));
    try testing.expectError(error.GroupDoesNotDivide, widen(a, h, .{ .bits = 8, .group = 128 }));
    try testing.expectError(error.UnsupportedQuantization, widen(a, h, .{ .bits = 7, .group = 64 }));
    try testing.expectError(error.UnsupportedQuantization, widen(a, h, .{ .bits = 8, .group = 16 }));
    try testing.expectError(error.UnsupportedQuantization, widen(a, h, .{ .bits = 8, .group = 48 }));
    var bad = h;
    bad.bits = 4;
    try testing.expectError(error.UnexpectedTensor, widen(a, bad, .{ .bits = 8, .group = 32 }));
}

test "the common width takes the most bits and the smallest group" {
    try testing.expectEqual(Width{ .bits = 5, .group = 32 }, common(&.{ .{ .bits = 4, .group = 64 }, .{ .bits = 5, .group = 128 }, .{ .bits = 2, .group = 32 } }));
    try testing.expectEqual(Width{ .bits = 6, .group = 32 }, common(&.{.{ .bits = 6, .group = 32 }}));
}

test "fused members keep each one's values in order at the common width" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0xf05e);
    const rng = prng.random();
    const k = 512;
    const members = [_]Host{
        try sample(a, rng, &.{4}, k, .{ .bits = 4, .group = 64 }),
        try sample(a, rng, &.{3}, k, .{ .bits = 5, .group = 128 }),
        try sample(a, rng, &.{5}, k, .{ .bits = 2, .group = 32 }),
    };
    const f = try fuse(a, &members);
    try testing.expectEqual(@as(u8, 5), f.bits);
    try testing.expectEqual(@as(u16, 32), f.group);
    try testing.expectEqual(@as(usize, 12), f.words.shape[0]);
    var want: std.ArrayList(f64) = .empty;
    for (members) |m| try want.appendSlice(a, try dequantized(a, m));
    try expectBits(want.items, try dequantized(a, f));
    const other = try sample(a, rng, &.{2}, 256, .{ .bits = 4, .group = 64 });
    try testing.expectError(error.UnexpectedTensor, fuse(a, &.{ members[0], other }));
}
