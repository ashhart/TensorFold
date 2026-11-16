//! Ling MLA RoPE: GPT-J pairs then halves, matching ling_forward.py rot() and HF apply_rotary_pos_emb_interleave.
const std = @import("std");
const config = @import("config.zig");
const bf16_round = @import("bf16_round");
const mla = @import("mla.zig");

pub const rope_dim: u32 = mla.rope_dim;
pub const nfreq: u32 = mla.nfreq;
pub const invFreq = mla.invFreq;
pub const rotate = mla.rotate;
pub const rotateRows = mla.rotateRows;

fn tinyTheta() !f64 {
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, @embedFile("fixtures/tiny_config.json"), &why);
    try std.testing.expectEqual(@as(u32, 64), c.rope_dim);
    try std.testing.expectEqual(config.RopeStyle.gptj, c.rope_style);
    try std.testing.expectEqual(@as(f64, 6000000), c.rope_theta);
    try std.testing.expectEqual(@as(u32, 192), c.qkDim());
    try std.testing.expectEqual(@as(f32, @floatCast(1.0 / @sqrt(@as(f64, 192.0)))), c.smScale());
    return c.rope_theta;
}

test "tiny MLA rope is 64 GPT-J columns, theta 6e6, softmax 1/sqrt(192)" {
    _ = try tinyTheta();
}

test "pos 0 writes GPT-J pairs as halves: the identity rotation" {
    const theta = try tinyTheta();
    var z: [rope_dim]f64 = undefined;
    for (&z, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i)) + 0.25;
    var out: [rope_dim]f64 = undefined;
    rotate(&out, &z, 0, theta);
    for (0..nfreq) |i| {
        try std.testing.expectEqual(z[2 * i], out[i]);
        try std.testing.expectEqual(z[2 * i + 1], out[nfreq + i]);
    }
}

test "pos 1, 8 and 1024 rotate the first pair by pos radians (frequency 0 is theta^0 = 1)" {
    const theta = try tinyTheta();
    try std.testing.expectEqual(@as(f64, 1), invFreq(theta, 0));
    var z: [rope_dim]f64 = @splat(0);
    z[0] = 1;
    z[1] = 0;
    for ([_]u32{ 1, 8, 1024 }) |pos| {
        var out: [rope_dim]f64 = undefined;
        rotate(&out, &z, pos, theta);
        const p: f64 = @floatFromInt(pos);
        try std.testing.expectApproxEqAbs(@cos(p), out[0], 1e-12);
        try std.testing.expectApproxEqAbs(@sin(p), out[nfreq], 1e-12);
        for (1..nfreq) |i| {
            try std.testing.expectEqual(@as(f64, 0), out[i]);
            try std.testing.expectEqual(@as(f64, 0), out[nfreq + i]);
        }
    }
}

test "a pos0 offset of 8 on four rows equals rotating each row at its absolute index" {
    const theta = try tinyTheta();
    const rows: u32 = 4;
    const pos0: u32 = 8;
    var z: [rows * rope_dim]f64 = undefined;
    for (&z, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i % 17)) * 0.125 - 1.0;
    var chunk: [rows * rope_dim]f64 = undefined;
    rotateRows(&chunk, &z, rows, pos0, theta);
    var fresh: [rows * rope_dim]f64 = undefined;
    for (0..rows) |t| {
        rotate(fresh[t * rope_dim ..][0..rope_dim], z[t * rope_dim ..][0..rope_dim], pos0 + @as(u32, @intCast(t)), theta);
    }
    for (&chunk, fresh) |a, b| try std.testing.expectEqual(a, b);
}

test "round32 keeps signalling NaN 0x7f800001 and all-ones 0x7fffffff as NaN" {
    const cases = [_]u32{ 0x7f800001, 0x7fffffff };
    for (cases) |u| {
        const v: f32 = @bitCast(u);
        try std.testing.expect(std.math.isNan(v));
        const h = bf16_round.bits(v);
        try std.testing.expectEqual(@as(u16, @truncate((u >> 16) | 0x40)), h);
        try std.testing.expect(std.math.isNan(bf16_round.value(h)));
        try std.testing.expect(std.math.isNan(bf16_round.round32(v)));
    }
}
