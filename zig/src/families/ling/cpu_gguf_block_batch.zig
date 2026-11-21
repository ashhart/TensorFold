//! Shared weight unpack for prompt rows over intact GGUF blocks.
const std = @import("std");
const act_mod = @import("cpu_gguf_sdot_act.zig");
const V = act_mod.V;
const I = act_mod.I;
const sdotAcc = act_mod.sdotAcc;

inline fn half(p: [*]const u8) f32 {
    return @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, p[0..2], .little))));
}

inline fn loadU8x16(p: [*]const u8) @Vector(16, u8) {
    return @as(*align(1) const [16]u8, @ptrCast(p)).*;
}

inline fn loadI8x16Aligned(p: [*]align(16) const i8) V {
    return @as(*align(16) const [16]i8, @ptrCast(p)).*;
}

inline fn q6kQuarter16(ql: @Vector(16, u8), qh: @Vector(16, u8), comptime quarter: usize) V {
    const mask4: @Vector(16, u8) = @splat(0x0F);
    const lo = if (quarter < 2) (ql & mask4) else (ql >> @as(@Vector(16, u8), @splat(4)));
    const hi = switch (quarter) {
        0 => (qh & @as(@Vector(16, u8), @splat(0x03))) << @as(@Vector(16, u8), @splat(4)),
        1 => (qh & @as(@Vector(16, u8), @splat(0x0C))) << @as(@Vector(16, u8), @splat(2)),
        2 => qh & @as(@Vector(16, u8), @splat(0x30)),
        3 => (qh & @as(@Vector(16, u8), @splat(0xC0))) >> @as(@Vector(16, u8), @splat(2)),
        else => unreachable,
    };
    return @as(V, @bitCast(lo | hi)) -% @as(V, @splat(32));
}

pub inline fn q4k(comptime M: usize, p: [*]const u8, acts: [*]const act_mod.Activation256, stride: usize) [M]f32 {
    const mask4: @Vector(16, u8) = @splat(0x0F);
    const shift4: @Vector(16, u8) = @splat(4);
    var result: [M]f32 = undefined;
    const d = half(p);
    const m = half(p + 2);
    const scales = p + 4;
    var sc: [8]i32 = undefined;
    var mn: [8]i32 = undefined;
    inline for (0..4) |grp| {
        sc[grp] = @intCast(scales[grp] & 63);
        mn[grp] = @intCast(scales[grp + 4] & 63);
    }
    inline for (4..8) |grp| {
        sc[grp] = @intCast((scales[grp + 4] & 15) | ((scales[grp - 4] >> 6) << 4));
        mn[grp] = @intCast((scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4));
    }
    var sum_vecs: [M]I = @splat(@splat(0));
    var minimum: [M]i32 = @splat(0);
    inline for (0..4) |chunk| {
        const qs = p + 16 + chunk * 32;
        const raw0 = loadU8x16(qs);
        const raw1 = loadU8x16(qs + 16);
        const lo0: V = @bitCast(raw0 & mask4);
        const lo1: V = @bitCast(raw1 & mask4);
        const hi0: V = @bitCast(raw0 >> shift4);
        const hi1: V = @bitCast(raw1 >> shift4);
        const g_lo = 2 * chunk;
        const g_hi = 2 * chunk + 1;
        const sc_lo_v: I = @splat(sc[g_lo]);
        const sc_hi_v: I = @splat(sc[g_hi]);
        inline for (0..M) |r| {
            const act = &acts[r * stride];
            const xc = @as([*]align(16) const i8, &act.values) + chunk * 64;
            var acc_lo: I = @splat(0);
            acc_lo = sdotAcc(acc_lo, lo0, loadI8x16Aligned(xc));
            acc_lo = sdotAcc(acc_lo, lo1, loadI8x16Aligned(xc + 16));
            sum_vecs[r] +%= acc_lo *% sc_lo_v;
            minimum[r] +%= mn[g_lo] *% act.sums[g_lo];

            var acc_hi: I = @splat(0);
            acc_hi = sdotAcc(acc_hi, hi0, loadI8x16Aligned(xc + 32));
            acc_hi = sdotAcc(acc_hi, hi1, loadI8x16Aligned(xc + 48));
            sum_vecs[r] +%= acc_hi *% sc_hi_v;
            minimum[r] +%= mn[g_hi] *% act.sums[g_hi];
        }
    }
    inline for (0..M) |r| {
        const scale = acts[r * stride].scale;
        const weighted: i32 = @reduce(.Add, sum_vecs[r]);
        result[r] = d * scale * @as(f32, @floatFromInt(weighted)) - m * scale * @as(f32, @floatFromInt(minimum[r]));
    }
    return result;
}

pub inline fn q6k(comptime M: usize, p: [*]const u8, acts: [*]const act_mod.Activation256, stride: usize) [M]f32 {
    var result: [M]f32 = undefined;
    const d = half(p + 208);
    var sum_vecs: [M]I = @splat(@splat(0));
    inline for (0..2) |h| {
        const ql_ptr = p + h * 64;
        const qh_ptr = p + 128 + h * 32;
        const sc_ptr = p + 192 + h * 8;
        const ql0 = loadU8x16(ql_ptr);
        const ql1 = loadU8x16(ql_ptr + 16);
        const ql2 = loadU8x16(ql_ptr + 32);
        const ql3 = loadU8x16(ql_ptr + 48);
        const qh0 = loadU8x16(qh_ptr);
        const qh1 = loadU8x16(qh_ptr + 16);
        inline for (0..4) |quarter| {
            const ql_a = if (quarter % 2 == 0) ql0 else ql2;
            const ql_b = if (quarter % 2 == 0) ql1 else ql3;
            const c_a = q6kQuarter16(ql_a, qh0, quarter);
            const c_b = q6kQuarter16(ql_b, qh1, quarter);
            const s_a: I = @splat(@as(i32, @as(i8, @bitCast(sc_ptr[quarter * 2]))));
            const s_b: I = @splat(@as(i32, @as(i8, @bitCast(sc_ptr[quarter * 2 + 1]))));
            inline for (0..M) |r| {
                const act = &acts[r * stride];
                const xq = @as([*]align(16) const i8, &act.values) + h * 128 + quarter * 32;
                sum_vecs[r] +%= sdotAcc(@splat(0), c_a, loadI8x16Aligned(xq)) *% s_a;
                sum_vecs[r] +%= sdotAcc(@splat(0), c_b, loadI8x16Aligned(xq + 16)) *% s_b;
            }
        }
    }
    inline for (0..M) |r| {
        const scale = acts[r * stride].scale;
        result[r] = d * scale * @as(f32, @floatFromInt(@reduce(.Add, sum_vecs[r])));
    }
    return result;
}
