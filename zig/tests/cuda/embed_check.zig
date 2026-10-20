//! MLX affine token rows against core/embed_ref.zig's float64 value at their rounding points, bit for bit.
const std = @import("std");
const core = @import("core");

const eref = core.embed_ref;
const bfr = core.bf16_round;

/// One call's host inputs: the table's words, scales and biases as bf16 bits, the ids, and the form.
pub const Case = struct {
    rows: usize,
    d: usize,
    bits: u4,
    group: usize,
    scale: ?f32,
    ids: []const i32,
    w: []const u32,
    s: []const u16,
    b: []const u16,
};

/// What a check saw: outputs bit-equal to float64's value, and NaN where float64's is NaN.
pub const Stats = struct { equal: usize = 0, nan: usize = 0 };

/// Element i of row `row`: float64's value at the rounding points.
pub fn want(k: Case, row: usize, i: usize) f64 {
    const t: usize = @intCast(k.ids[row]);
    const words = k.d / 32 * k.bits;
    const q = eref.code(k.w[t * words ..][0..words], i, k.bits);
    const g = t * (k.d / k.group) + i / k.group;
    return eref.affine(q, bfr.value(k.s[g]), bfr.value(k.b[g]), if (k.scale) |x| x else null);
}

/// Every output's bits equal float64's value rounded at the points (NaN where it is NaN, any payload).
pub fn verify(k: Case, out: []const u16, stats: *Stats, msg: []u8) !?[]const u8 {
    for (0..k.rows) |row| for (0..k.d) |i| {
        const w = want(k, row, i);
        const got = out[row * k.d + i];
        const ok = if (std.math.isNan(w)) std.math.isNan(bfr.value(got)) else got == bfr.bits(@floatCast(w));
        if (!ok) return try std.fmt.bufPrint(msg, "row {d} out[{d}] = {e} where float64 at the rounding points gives {e}", .{ row, i, bfr.value(got), w });
        if (std.math.isNan(w)) stats.nan += 1 else stats.equal += 1;
    };
    return null;
}

/// MLX's words of `codes` `bits` wide, low code first.
pub fn pack(a: std.mem.Allocator, codes: []const u32, bits: u4) ![]u32 {
    const w = try a.alloc(u32, codes.len * bits / 32);
    @memset(w, 0);
    for (codes, 0..) |q, i| {
        const bit = i * bits;
        w[bit / 32] |= q << @intCast(bit % 32);
        if (bit % 32 + bits > 32) w[bit / 32 + 1] |= q >> @intCast(32 - bit % 32);
    }
    return w;
}

test "pack is embed_ref.code's inverse" {
    var prng = std.Random.DefaultPrng.init(4);
    var codes: [192]u32 = undefined;
    for ([_]u4{ 2, 3, 5, 6, 8 }) |bits| {
        for (&codes) |*q| q.* = prng.random().int(u32) & ((@as(u32, 1) << bits) - 1);
        const w = try pack(std.testing.allocator, &codes, bits);
        defer std.testing.allocator.free(w);
        for (codes, 0..) |q, i| try std.testing.expectEqual(q, eref.code(w, i, bits));
    }
}
