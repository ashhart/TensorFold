//! Shared activation quantizers and SDOT helper for Ling fast-mode GGUF projections.
const std = @import("std");

pub const V = @Vector(16, i8);
pub const I = @Vector(4, i32);

extern fn @"llvm.aarch64.neon.sdot.v4i32.v16i8"(I, V, V) I;

pub const Activation256 = struct {
    values: [256]i8 align(16),
    sums: [8]i32 align(16),
    bsums: [16]i16 align(16),
    scale: f32,
};

pub const Activation32 = struct {
    values: [32]i8 align(16),
    scale: f32,
};

pub inline fn sdotAcc(acc: I, a: V, b: V) I {
    if (@import("builtin").cpu.arch == .aarch64) {
        return @"llvm.aarch64.neon.sdot.v4i32.v16i8"(acc, a, b);
    }
    const prod = @as(@Vector(16, i32), a) *% @as(@Vector(16, i32), b);
    return acc +% I{
        prod[0] +% prod[1] +% prod[2] +% prod[3],
        prod[4] +% prod[5] +% prod[6] +% prod[7],
        prod[8] +% prod[9] +% prod[10] +% prod[11],
        prod[12] +% prod[13] +% prod[14] +% prod[15],
    };
}

pub fn checkTarget() !void {
    if (comptime @import("builtin").cpu.arch == .aarch64 and !std.Target.aarch64.featureSetHas(@import("builtin").cpu.features, .dotprod)) {
        return error.DotprodTargetRequired;
    }
}

pub fn quantize256(x: []const f32, acts: []Activation256, comptime with_sums: bool) !void {
    const F8 = @Vector(8, f32);
    const U8 = @Vector(8, u32);
    const lo_clamp: F8 = @splat(-127.0);
    const hi_clamp: F8 = @splat(127.0);
    const zero_v: F8 = @splat(0.0);
    const exp_mask: U8 = @splat(0x7F80_0000);

    for (acts, 0..) |*act, b| {
        const base = x.ptr + b *% 256;
        var vmax: F8 = zero_v;
        var bad: U8 = @splat(0);
        var i: usize = 0;
        while (i < 256) : (i +%= 8) {
            const v: F8 = @as(*align(1) const [8]f32, @ptrCast(base + i)).*;
            const bits: U8 = @bitCast(v);
            bad |= @select(u32, (bits & exp_mask) == exp_mask, @as(U8, @splat(1)), @as(U8, @splat(0)));
            vmax = @max(vmax, @abs(v));
        }
        if (@reduce(.Or, bad) != 0) return error.NonfiniteInput;
        const maximum = @reduce(.Max, vmax);
        if (!std.math.isFinite(maximum)) return error.NonfiniteInput;
        act.scale = maximum / 127;
        const inverse: f32 = if (maximum == 0) 0 else 127 / maximum;
        if (!std.math.isFinite(inverse) or (maximum != 0 and act.scale == 0)) return error.ActivationScaleRange;
        const inv_v: F8 = @splat(inverse);

        inline for (0..8) |grp| {
            var bsum0: i32 = 0;
            var bsum1: i32 = 0;
            inline for (0..4) |sub| {
                const off = grp * 32 + sub * 8;
                const v: F8 = @as(*align(1) const [8]f32, @ptrCast(base + off)).*;
                const scaled = std.math.clamp(v * inv_v, lo_clamp, hi_clamp);
                const q8: @Vector(8, i8) = @intFromFloat(@round(scaled));
                @as(*align(8) [8]i8, @ptrCast(&act.values[off])).* = q8;
                if (with_sums) {
                    const s8 = @reduce(.Add, @as(@Vector(8, i32), q8));
                    if (sub < 2) bsum0 +%= s8 else bsum1 +%= s8;
                }
            }
            if (with_sums) {
                act.bsums[grp * 2 + 0] = @intCast(bsum0);
                act.bsums[grp * 2 + 1] = @intCast(bsum1);
                act.sums[grp] = bsum0 +% bsum1;
            }
        }
    }
}

pub fn quantize32(x: []const f32, acts: []Activation32) !void {
    const F8 = @Vector(8, f32);
    const U8 = @Vector(8, u32);
    const lo_clamp: F8 = @splat(-127.0);
    const hi_clamp: F8 = @splat(127.0);
    const exp_mask: U8 = @splat(0x7F80_0000);

    for (acts, 0..) |*act, b| {
        const base = x.ptr + b *% 32;
        const v0: F8 = @as(*align(1) const [8]f32, @ptrCast(base)).*;
        const v1: F8 = @as(*align(1) const [8]f32, @ptrCast(base + 8)).*;
        const v2: F8 = @as(*align(1) const [8]f32, @ptrCast(base + 16)).*;
        const v3: F8 = @as(*align(1) const [8]f32, @ptrCast(base + 24)).*;
        const b0: U8 = @bitCast(v0);
        const b1: U8 = @bitCast(v1);
        const b2: U8 = @bitCast(v2);
        const b3: U8 = @bitCast(v3);
        const bad = @select(u32, (b0 & exp_mask) == exp_mask, @as(U8, @splat(1)), @as(U8, @splat(0))) |
            @select(u32, (b1 & exp_mask) == exp_mask, @as(U8, @splat(1)), @as(U8, @splat(0))) |
            @select(u32, (b2 & exp_mask) == exp_mask, @as(U8, @splat(1)), @as(U8, @splat(0))) |
            @select(u32, (b3 & exp_mask) == exp_mask, @as(U8, @splat(1)), @as(U8, @splat(0)));
        if (@reduce(.Or, bad) != 0) return error.NonfiniteInput;
        const maximum = @reduce(.Max, @max(@max(@abs(v0), @abs(v1)), @max(@abs(v2), @abs(v3))));
        if (!std.math.isFinite(maximum)) return error.NonfiniteInput;
        act.scale = maximum / 127;
        const inverse: f32 = if (maximum == 0) 0 else 127 / maximum;
        if (!std.math.isFinite(inverse) or (maximum != 0 and act.scale == 0)) return error.ActivationScaleRange;
        const inv_v: F8 = @splat(inverse);
        const q0: @Vector(8, i8) = @intFromFloat(@round(std.math.clamp(v0 * inv_v, lo_clamp, hi_clamp)));
        const q1: @Vector(8, i8) = @intFromFloat(@round(std.math.clamp(v1 * inv_v, lo_clamp, hi_clamp)));
        const q2: @Vector(8, i8) = @intFromFloat(@round(std.math.clamp(v2 * inv_v, lo_clamp, hi_clamp)));
        const q3: @Vector(8, i8) = @intFromFloat(@round(std.math.clamp(v3 * inv_v, lo_clamp, hi_clamp)));
        @as(*align(8) [8]i8, @ptrCast(&act.values[0])).* = q0;
        @as(*align(8) [8]i8, @ptrCast(&act.values[8])).* = q1;
        @as(*align(8) [8]i8, @ptrCast(&act.values[16])).* = q2;
        @as(*align(8) [8]i8, @ptrCast(&act.values[24])).* = q3;
    }
}
