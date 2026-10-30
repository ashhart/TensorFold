//! Independent numerical witnesses for paged Qwen attention, using the original CPU caches as the oracle.
const std = @import("std");
const core = @import("core");
const c = core.contract.spec;
const rows = core.contract.rows;
const hd = 256;
const q_heads = 24;
const kv_heads = 4;
const layers = 2;

pub const Mode = enum { random, stress, nonfinite };

fn depth(shape: c.Shape) usize {
    return shape.k;
}

const sum_spec: c.Spec = .{
    .name = "cuda Qwen page attention FP32 value sums",
    .accum = .f32,
    .depth = depth,
    .ops = 1,
    .out = .bf16,
    .served = &.{.{ .rows = 1, .n = q_heads, .k = 8193 }},
};

fn stressValue(pos: usize) f64 {
    return switch (pos % 4) {
        0 => 4096,
        1, 3 => 1,
        2 => -4096,
        else => unreachable,
    };
}

fn scaleAt(cache: []u8, off: usize, scale: f32) void {
    std.mem.writeInt(u32, cache[off..][0..4], @bitCast(scale), .little);
}

fn put(f: anytype, cache: []u8, head: usize, pos: usize, d: usize, value: f32, code: u8, scale: f32) void {
    const sp = f.logical(1, 1, head, pos);
    switch (f.format) {
        .bf16 => std.mem.writeInt(u16, cache[sp[0].off + 2 * d ..][0..2], c.toBf16(value), .little),
        .fp8 => {
            cache[sp[0].off + d] = code;
            scaleAt(cache, sp[1].off, scale);
        },
    }
}

/// Zero scores and new values; all expected nonzero/non-finite output data comes from the cached CPU prefix.
pub fn fill(f: anytype, mode: Mode) !void {
    if (mode == .random) return;
    @memset(f.q, 0);
    @memset(f.k, 0);
    @memset(f.v, 0);
    for (f.kv) |cache| {
        @memset(cache, 0);
        if (f.format == .fp8) for (0..layers) |layer| for (0..2) |part| for (0..kv_heads) |head| for (0..f.cap) |pos| {
            scaleAt(cache, f.logical(layer, part, head, pos)[1].off, 1);
        };
    }
    for (0..2) |st| {
        if (mode == .stress) {
            if (f.c.rows != 1 or f.c.lens[st] == 0 or f.c.lens[st] % 4 != 0 or f.c.lens[st] > 8192)
                return error.InvalidQwenSumFixture;
            const codes = [_]u8{ 0x68, 0x08, 0xe8, 0x08 };
            for (0..kv_heads) |head| for (0..f.c.lens[st]) |pos| {
                put(f.*, f.kv[st], head, pos, 0, @floatCast(stressValue(pos)), codes[pos % 4], 64);
            };
        } else {
            if (f.c.lens[st] < 2) return error.InvalidQwenNonfiniteFixture;
            put(f.*, f.kv[st], 0, 0, 0, std.math.nan(f32), 0x7f, 1);
            if (f.format == .bf16) {
                put(f.*, f.kv[st], 1, 0, 0, std.math.inf(f32), 0, 1);
                put(f.*, f.kv[st], 2, 0, 0, -std.math.inf(f32), 0, 1);
                put(f.*, f.kv[st], 3, 0, 0, std.math.inf(f32), 0, 1);
                put(f.*, f.kv[st], 3, 1, 0, -std.math.inf(f32), 0, 1);
            } else {
                for (0..hd) |d| {
                    put(f.*, f.kv[st], 1, 0, d, 0, 0x38, std.math.inf(f32));
                    put(f.*, f.kv[st], 2, 0, d, 0, 0xb8, std.math.inf(f32));
                    put(f.*, f.kv[st], 3, 0, d, 0, 0x38, std.math.inf(f32));
                    put(f.*, f.kv[st], 3, 1, d, 0, 0xb8, std.math.inf(f32));
                }
            }
        }
    }
}

/// Every registered accumulator order must retain the exact FP32 result and expose both narrower accumulators.
fn stressProof(terms: []const f64, ones: []const f64, want: f64) bool {
    const bound = c.boundExact(sum_spec, want);
    for ([_]rows.Order{ .sequential, .pairwise, .strided }) |order| {
        const good = rows.sumIn(.f32, order, terms) / rows.sumIn(.f32, order, ones);
        if (!std.math.isFinite(good) or !(@abs(good - want) <= bound)) return false;
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            const bad = rows.sumIn(acc, order, terms) / rows.sumIn(acc, order, ones);
            if (std.math.isFinite(bad) and !(@abs(bad - want) >= 100 * bound)) return false;
        }
    }
    return true;
}

fn outputValue(out: []const u8, row: usize, head: usize, d: usize) f64 {
    return c.read(.bf16, out, (row * q_heads + head) * hd + d);
}

fn near(got: f64, want: f64) bool {
    return c.classOf(got) == c.classOf(want) and
        (!std.math.isFinite(want) or @abs(got - want) <= c.boundExact(sum_spec, want));
}

/// The original prefix and the known-zero projected rows define the oracle, never the GPU cache readback.
pub fn agrees(a: std.mem.Allocator, f: anytype, result: anytype, mode: Mode) !bool {
    if (mode == .random) return true;
    for ([_][]const u16{ f.q, f.k, f.v }) |input| for (input) |v| if (c.fromBf16(v) != 0) return error.InvalidQwenSumFixture;
    if (result.out.len != 2 * f.c.rows * q_heads * hd * 2 or result.q.len != f.q.len * 2) return false;
    for (0..result.q.len / 2) |i| if (c.read(.bf16, result.q, i) != 0) return false;
    const most = @max(f.c.lens[0], f.c.lens[1]) + f.c.rows;
    const terms = try a.alloc(f64, most);
    defer a.free(terms);
    const ones = try a.alloc(f64, most);
    defer a.free(ones);
    @memset(ones, 1);
    var finite: usize = 0;
    var nonfinite: usize = 0;
    for (0..2) |st| for (0..f.c.rows) |r| for (0..kv_heads) |kh| {
        const n = f.c.lens[st] + r + 1;
        for (0..hd) |d| {
            var total: f64 = 0;
            for (terms[0..n], 0..) |*t, pos| {
                t.* = if (pos < f.c.lens[st]) f.value(f.kv[st], 1, 1, kh, pos, d) else 0;
                total += t.*;
            }
            const want = total / @as(f64, @floatFromInt(n));
            if (mode == .stress and d == 0 and !stressProof(terms[0..n], ones[0..n], want)) return false;
            if (std.math.isFinite(want)) finite += 1 else nonfinite += 1;
            for (0..q_heads / kv_heads) |local_head| {
                const got = outputValue(result.out, st * f.c.rows + r, kh * (q_heads / kv_heads) + local_head, d);
                if (!near(got, want)) return false;
            }
        }
    };
    return finite > 0 and (mode != .nonfinite or nonfinite > 0);
}

test "Qwen page attention stress rejects FP16 and BF16 under all three accumulator orders" {
    const a = std.testing.allocator;
    for ([_]usize{ 4096, 8192 }) |cached| {
        const terms = try a.alloc(f64, cached + 1);
        defer a.free(terms);
        const ones = try a.alloc(f64, cached + 1);
        defer a.free(ones);
        var total: f64 = 0;
        var positive: f64 = 0;
        for (terms[0..cached], 0..) |*t, pos| {
            t.* = stressValue(pos);
            total += t.*;
            positive += @max(t.*, 0);
        }
        terms[cached] = 0;
        @memset(ones, 1);
        try std.testing.expect(positive < 0x1p24);
        try std.testing.expect(stressProof(terms, ones, total / @as(f64, @floatFromInt(cached + 1))));
    }
}

test "Qwen page FP8 stress codes and scale recover exact integer terms" {
    const codes = [_]u8{ 0x68, 0x08, 0xe8, 0x08 };
    for (codes, 0..) |code, pos| {
        try std.testing.expectEqual(stressValue(pos), core.quant.fp8.e4m3(code) * 64);
        try std.testing.expectEqual(stressValue(pos), @as(f64, c.fromBf16(c.toBf16(@floatCast(stressValue(pos))))));
    }
}

test "Qwen page numerical class check rejects lost NaN, infinity and a corrupted finite row" {
    try std.testing.expect(near(std.math.nan(f64), std.math.nan(f64)));
    try std.testing.expect(!near(0, std.math.nan(f64)));
    try std.testing.expect(!near(-std.math.inf(f64), std.math.inf(f64)));
    try std.testing.expect(!near(0, 1));
    try std.testing.expect(!near(std.math.inf(f64), 0));
    try std.testing.expect(near(0, 0));
}

const CpuFixture = struct {
    const Format = enum { bf16, fp8 };
    const Span = struct { off: usize, len: usize };
    c: struct { lens: [2]usize = .{ 8, 16 }, rows: usize = 5 } = .{},
    format: Format,
    cap: usize = 32,
    kv: [2][]u8,
    q: []u16,
    k: []u16,
    v: []u16,

    fn logical(f: CpuFixture, layer: usize, part: usize, head: usize, pos: usize) [2]Span {
        const row: usize = if (f.format == .bf16) hd * 2 else hd;
        const at = layer * 2 * kv_heads * f.cap * @as(usize, if (f.format == .bf16) hd * 2 else hd + 4);
        const blk = part * kv_heads + head;
        return .{ .{ .off = at + (blk * f.cap + pos) * row, .len = row }, .{ .off = at + 2 * kv_heads * f.cap * hd + (blk * f.cap + pos) * 4, .len = if (f.format == .fp8) 4 else 0 } };
    }

    fn value(f: CpuFixture, cache: []const u8, layer: usize, part: usize, head: usize, pos: usize, d: usize) f64 {
        const sp = f.logical(layer, part, head, pos);
        return if (f.format == .bf16) c.read(.bf16, cache[sp[0].off..], d) else core.quant.fp8.e4m3(cache[sp[0].off + d]) * @as(f64, @as(f32, @bitCast(std.mem.readInt(u32, cache[sp[1].off..][0..4], .little))));
    }
};

test "Qwen non-finite fixture keeps finite heads and rejects planted output classes in both formats" {
    const a = std.testing.allocator;
    for (std.enums.values(CpuFixture.Format)) |format| {
        const bytes = layers * 2 * kv_heads * 32 * @as(usize, if (format == .bf16) hd * 2 else hd + 4);
        const first = try a.alloc(u8, bytes);
        defer a.free(first);
        const second = try a.alloc(u8, bytes);
        defer a.free(second);
        const q = try a.alloc(u16, 2 * 5 * q_heads * hd);
        defer a.free(q);
        const k = try a.alloc(u16, 2 * 5 * kv_heads * hd);
        defer a.free(k);
        const v = try a.alloc(u16, k.len);
        defer a.free(v);
        var f: CpuFixture = .{ .format = format, .kv = .{ first, second }, .q = q, .k = k, .v = v };
        try fill(&f, .nonfinite);
        const out = try a.alloc(u8, q.len * 2);
        defer a.free(out);
        for (0..2 * 5) |row| for (0..q_heads) |head| for (0..hd) |d| {
            const kh = head / (q_heads / kv_heads);
            const want: f32 = if (d != 0 and (format == .bf16 or kh == 0)) 0 else switch (kh) {
                0, 3 => std.math.nan(f32),
                1 => std.math.inf(f32),
                2 => -std.math.inf(f32),
                else => unreachable,
            };
            std.mem.writeInt(u16, out[2 * ((row * q_heads + head) * hd + d) ..][0..2], c.toBf16(want), .little);
        };
        const result = .{ .q = std.mem.sliceAsBytes(q), .out = out };
        try std.testing.expect(try agrees(a, f, result, .nonfinite));
        std.mem.writeInt(u16, out[0..2], 0, .little);
        try std.testing.expect(!try agrees(a, f, result, .nonfinite));
        std.mem.writeInt(u16, out[0..2], c.toBf16(std.math.nan(f32)), .little);
        std.mem.writeInt(u16, out[2..4], c.toBf16(1), .little);
        try std.testing.expect(!try agrees(a, f, result, .nonfinite));
    }
}
