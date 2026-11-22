//! The attention's host models against the oracle's dump (tiny-random/inkling-2's bias tables, no GPU).
const std = @import("std");
const attn = @import("attn.zig");
const order = @import("attn_order.zig");
const bf16_round = @import("bf16_round");

const t = std.testing;
const dump = @embedFile("fixtures/attn_tiny.bin");

pub const Case = struct {
    kind: u32, // 0 sliding, 1 full
    shape: attn.Shape,
    in: attn.Inputs,
    y: []f64, // float64 output
    torch16: []u16, // the transformers functions' bf16 output bits
};

fn at(comptime T: type, off: usize) T {
    return std.mem.readInt(T, dump[off..][0..@sizeOf(T)], .little);
}

/// `n` bf16 values widened to fp32.
fn widen(a: std.mem.Allocator, off: *usize, n: usize) ![]f32 {
    const out = try a.alloc(f32, n);
    for (out) |*v| {
        v.* = bf16_round.value(at(u16, off.*));
        off.* += 2;
    }
    return out;
}

pub fn cases(a: std.mem.Allocator) ![]Case {
    try t.expect(std.mem.eql(u8, dump[0..8], "TFINKAT1"));
    const list = try a.alloc(Case, at(u32, 8));
    var off: usize = 12;
    for (list) |*c| {
        c.kind = at(u32, off);
        const f: [8]usize = blk: {
            var x: [8]usize = undefined;
            for (&x, 0..) |*v, i| v.* = at(u32, off + 4 + 4 * i);
            break :blk x;
        };
        c.shape = .{ .heads = f[0], .kv_heads = f[1], .hd = f[2], .drel = f[3], .rext = f[4], .window = f[5], .rows = f[6], .pos0 = f[7] };
        off += 36;
        const s = c.shape;
        const out = s.rows * s.heads * s.hd;
        const pos = s.pos0 + s.rows;
        c.in.q = try widen(a, &off, out);
        c.in.k = try widen(a, &off, pos * s.kv_heads * s.hd);
        c.in.v = try widen(a, &off, pos * s.kv_heads * s.hd);
        c.in.rel = try widen(a, &off, s.rows * s.heads * s.drel);
        c.in.proj = try widen(a, &off, s.drel * s.rext);
        c.in.sink = try widen(a, &off, s.rows * s.heads);
        c.in.gate = try widen(a, &off, out);
        c.y = try a.alloc(f64, out);
        for (c.y) |*v| {
            v.* = @bitCast(at(u64, off));
            off += 8;
        }
        c.torch16 = try a.alloc(u16, out);
        for (c.torch16) |*v| {
            v.* = at(u16, off);
            off += 2;
        }
    }
    try t.expectEqual(dump.len, off);
    return list;
}

/// The least bound / |e| over every head, e = got - y, bound = (|y| + the head's max |y|) 2^-k; inf where every e is 0.
fn margin(c: Case, got: []const f64, comptime k: comptime_int) f64 {
    var least = std.math.inf(f64);
    const hd = c.shape.hd;
    for (0..c.shape.rows * c.shape.heads) |h| {
        const want = c.y[h * hd ..][0..hd];
        var top: f64 = 0;
        for (want) |v| top = @max(top, @abs(v));
        for (want, got[h * hd ..][0..hd]) |w, v| {
            const e = @abs(v - w);
            if (e != 0) least = @min(least, (@abs(w) + top) * std.math.pow(f64, 2, -k) / e);
        }
    }
    return least;
}

test "the inputs are bf16 values and the fixture holds both kinds, the seams and the checkpoint's tables" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const list = try cases(arena.allocator());
    var sliding: usize = 0;
    for (list) |c| {
        sliding += @intFromBool(c.kind == 0);
        try attn.shaped(c.shape, c.in, c.y.len);
        const arrays = [_][]const f32{ c.in.q, c.in.k, c.in.v, c.in.rel, c.in.proj, c.in.sink, c.in.gate };
        for (arrays) |x| for (x) |v| try t.expectEqual(@as(u32, 0), @as(u32, @bitCast(v)) & 0xffff);
    }
    try t.expectEqual(@as(usize, 6), list.len);
    try t.expectEqual(@as(usize, 3), sliding);
    // the window of 64 reaches its edge inside the rows (positions 62 to 65); one case starts at position 0
    try t.expectEqual(@as(usize, 64), list[2].shape.window);
    try t.expectEqual(@as(usize, 62), list[2].shape.pos0);
    try t.expectEqual(@as(usize, 0), list[4].shape.pos0);
}

test "the float64 model equals the oracle's dump to 2^-40, per element" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (try cases(a)) |c| {
        const got = try a.alloc(f64, c.y.len);
        try attn.attend(f64, false, c.shape, c.in, got);
        const m = margin(c, got, 40);
        std.debug.print("attn f64 vs dump 2^-40 margin {d:.2}x (kind {d}, {d} rows, {d} heads)\n", .{ m, c.kind, c.shape.rows, c.shape.heads });
        try t.expect(m >= 1);
    }
}

test "eager fidelity: the eager-point model gives transformers' bf16 bits" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var total: usize = 0;
    var equal: usize = 0;
    var worst: u32 = 0; // in bf16 steps
    for (try cases(a)) |c| {
        const y32 = try a.alloc(f32, c.y.len);
        try attn.attend(f32, true, c.shape, c.in, y32);
        var same: usize = 0;
        for (y32, c.torch16) |v, want| {
            const raw: u32 = @bitCast(v);
            try t.expectEqual(@as(u32, 0), raw & 0xffff); // the model rounds its output to bf16 itself
            const bits: u16 = @intCast(raw >> 16);
            same += @intFromBool(bits == want);
            const step = @abs(@as(i32, bits) - @as(i32, want));
            worst = @max(worst, @as(u32, @intCast(step)));
        }
        std.debug.print("attn bf16 bits: {d} of {d} equal the module's (kind {d}, {d} rows)\n", .{ same, y32.len, c.kind, c.shape.rows });
        total += y32.len;
        equal += same;
    }
    std.debug.print("attn bf16 bits: {d} of {d} equal, worst {d} steps apart\n", .{ equal, total, worst });
    // a libm with another last bit in exp may flip a rounding: a thousandth may differ by one step
    try t.expect(equal * 1000 >= total * 999);
    try t.expect(worst <= 1);
}

test "bf16 bits: a tie goes to the even neighbour, above it up, infinity stays and a NaN stays a NaN" {
    const one: u32 = @bitCast(@as(f32, 1.0));
    try t.expectEqual(@as(u16, 0x3f80), bf16_round.bits(@bitCast(one + 0x8000))); // halfway from 1 to the next: even
    try t.expectEqual(@as(u16, 0x3f81), bf16_round.bits(@bitCast(one + 0x8001))); // just above it
    try t.expectEqual(@as(u16, 0x3f82), bf16_round.bits(@bitCast(one + 0x10000 + 0x8000))); // halfway: the even 0x3f82
    try t.expectEqual(@as(u16, 0x7f80), bf16_round.bits(std.math.inf(f32)));
    try t.expect(std.math.isNan(bf16_round.value(bf16_round.bits(std.math.nan(f32)))));
}

/// A bf16 value's place among all bf16 values in order, so a distance counts the values between two.
fn place(bits: u16) i32 {
    return if (bits & 0x8000 != 0) 0x8000 - @as(i32, bits & 0x7fff) else 0x8000 + @as(i32, bits);
}

test "eager fidelity: the kernels' order against transformers' bf16 bits, reported" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var total: usize = 0;
    var equal: usize = 0;
    var one: usize = 0;
    var worst: u32 = 0; // in bf16 values between
    var rel: f64 = 0; // the worst |got - transformers| / (|transformers| + the head's largest)
    for (try cases(a)) |c| {
        const y32 = try a.alloc(f32, c.y.len);
        try order.attendOrder(c.shape, c.in, y32, .{});
        const hd = c.shape.hd;
        for (0..c.shape.rows * c.shape.heads) |h| {
            var top: f64 = 0;
            for (c.torch16[h * hd ..][0..hd]) |w| top = @max(top, @abs(bf16_round.value(w)));
            for (y32[h * hd ..][0..hd], c.torch16[h * hd ..][0..hd]) |v, w| {
                const step = @abs(place(bf16_round.bits(v)) - place(w));
                equal += @intFromBool(step == 0);
                one += @intFromBool(step == 1);
                worst = @max(worst, @as(u32, @intCast(step)));
                rel = @max(rel, @abs(@as(f64, v) - bf16_round.value(w)) / (@abs(@as(f64, bf16_round.value(w))) + top));
            }
        }
        total += y32.len;
    }
    std.debug.print("attn order vs transformers bf16 bits: {d} of {d} equal, {d} one step apart, worst {d} steps, worst relative {e:.2}\n", .{ equal, total, one, worst, rel });
}
