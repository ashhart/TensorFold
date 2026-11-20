//! Multi-row 8x4 SDOT kernels for fast-mode prompt prefill; byte-identical to the 1-row repack8 kernels.
const std = @import("std");
const core = @import("core");
const g = core.gguf;
const sdot = @import("cpu_gguf_q4_sdot.zig");
const act_mod = @import("cpu_gguf_sdot_act.zig");
const repack8 = @import("cpu_gguf_repack8.zig");
const Pool = @import("cpu_gguf_pool.zig").Pool;

const V = act_mod.V;
const I = act_mod.I;
const F4 = @Vector(4, f32);

/// Multi-row Q4_K_8x4 block dot: unpacks each weight nibble vector once and reuses it across M prompt rows.
pub inline fn dotQ4Kx8BlockN(
    comptime M: usize,
    blk: *const repack8.BlockQ4Kx8,
    acts: [M]*const sdot.Activation256,
    acc_f32: *[M][2]F4,
) void {
    const mask4: @Vector(16, u8) = @splat(0x0F);
    const shift4: @Vector(16, u8) = @splat(4);

    const q4_d_0: F4 = @floatCast(@as(@Vector(4, f16), blk.d[0..4].*));
    const q4_d_1: F4 = @floatCast(@as(@Vector(4, f16), blk.d[4..8].*));
    const q4_m_0: F4 = @floatCast(@as(@Vector(4, f16), blk.m[0..4].*));
    const q4_m_1: F4 = @floatCast(@as(@Vector(4, f16), blk.m[4..8].*));

    var sum_acc: [M][2]I = @splat(.{ @splat(0), @splat(0) });
    var bias_acc: [M][2]I = @splat(.{ @splat(0), @splat(0) });
    var x_vecs: [M][*]const V = undefined;
    inline for (0..M) |r| x_vecs[r] = @ptrCast(@alignCast(&acts[r].values));
    const qs_vecs: [*]const @Vector(16, u8) = @ptrCast(@alignCast(&blk.qs));

    for (0..4) |sb| {
        var q8_0: [M]V = undefined;
        var q8_1: [M]V = undefined;
        var q8_2: [M]V = undefined;
        var q8_3: [M]V = undefined;
        inline for (0..M) |r| {
            q8_0[r] = x_vecs[r][sb * 4];
            q8_1[r] = x_vecs[r][sb * 4 + 1];
            q8_2[r] = x_vecs[r][sb * 4 + 2];
            q8_3[r] = x_vecs[r][sb * 4 + 3];
        }

        const q4_base = qs_vecs + sb * 16;
        const sc_lo = repack8.loadU8Pair(&blk.sc[2 * sb]);
        const sc_hi = repack8.loadU8Pair(&blk.sc[2 * sb + 1]);

        inline for (0..2) |cg| {
            var acc_lo: [M]I = @splat(@splat(0));
            var acc_hi: [M]I = @splat(@splat(0));
            inline for (0..4) |i| {
                const raw = q4_base[i * 2 + cg];
                const q_lo: V = @bitCast(raw & mask4);
                const q_hi: V = @bitCast(raw >> shift4);
                inline for (0..M) |r| {
                    acc_lo[r] = repack8.sdotLane(acc_lo[r], q_lo, q8_0[r], @intCast(i));
                    acc_hi[r] = repack8.sdotLane(acc_hi[r], q_hi, q8_2[r], @intCast(i));
                }
            }
            inline for (0..4) |i| {
                const raw = q4_base[(i + 4) * 2 + cg];
                const q_lo: V = @bitCast(raw & mask4);
                const q_hi: V = @bitCast(raw >> shift4);
                inline for (0..M) |r| {
                    acc_lo[r] = repack8.sdotLane(acc_lo[r], q_lo, q8_1[r], @intCast(i));
                    acc_hi[r] = repack8.sdotLane(acc_hi[r], q_hi, q8_3[r], @intCast(i));
                }
            }
            const s_lo: I = if (cg == 0) sc_lo.lo else sc_lo.hi;
            const s_hi: I = if (cg == 0) sc_hi.lo else sc_hi.hi;
            inline for (0..M) |r| {
                sum_acc[r][cg] +%= acc_lo[r] *% s_lo +% acc_hi[r] *% s_hi;
            }
        }

        const mn_lo = repack8.loadU8Pair(&blk.mn[2 * sb]);
        const mn_hi = repack8.loadU8Pair(&blk.mn[2 * sb + 1]);
        inline for (0..M) |r| {
            const sum_lo: I = @splat(acts[r].sums[2 * sb]);
            const sum_hi: I = @splat(acts[r].sums[2 * sb + 1]);
            bias_acc[r][0] +%= @as(I, mn_lo.lo) *% sum_lo +% @as(I, mn_hi.lo) *% sum_hi;
            bias_acc[r][1] +%= @as(I, mn_lo.hi) *% sum_lo +% @as(I, mn_hi.hi) *% sum_hi;
        }
    }

    inline for (0..M) |r| {
        const q8_d: F4 = @splat(acts[r].scale);
        const sb_scale_0 = q4_d_0 * q8_d;
        const sb_scale_1 = q4_d_1 * q8_d;
        const sb_min_0 = q4_m_0 * q8_d;
        const sb_min_1 = q4_m_1 * q8_d;
        acc_f32[r][0] += sb_scale_0 * @as(F4, @floatFromInt(sum_acc[r][0])) - sb_min_0 * @as(F4, @floatFromInt(bias_acc[r][0]));
        acc_f32[r][1] += sb_scale_1 * @as(F4, @floatFromInt(sum_acc[r][1])) - sb_min_1 * @as(F4, @floatFromInt(bias_acc[r][1]));
    }
}

inline fn dotQ6Kx8SubblockN(
    comptime M: usize,
    comptime sb_hi: bool,
    ql_vecs: [*]const @Vector(16, u8),
    qh_vecs: [*]const @Vector(16, u8),
    q8_base_l: [M][*]const i8,
    scales_l: [*]const i8,
    scales_h: [*]const i8,
    acc: *[M][2]I,
) void {
    const mask_lo: @Vector(16, u8) = @splat(0x03);
    const mask_hi: @Vector(16, u8) = @splat(0x30);
    const shift2: @Vector(16, u8) = @splat(2);
    const shift4: @Vector(16, u8) = @splat(4);

    var sb_acc_l0: [M]I = @splat(@splat(0));
    var sb_acc_h0: [M]I = @splat(@splat(0));
    var sb_acc_l1: [M]I = @splat(@splat(0));
    var sb_acc_h1: [M]I = @splat(@splat(0));

    inline for (0..4) |chunk| {
        var q8_l: [M]V = undefined;
        var q8_h: [M]V = undefined;
        inline for (0..M) |r| {
            q8_l[r] = repack8.loadDupI8x4(q8_base_l[r] + chunk * 4);
            q8_h[r] = repack8.loadDupI8x4(q8_base_l[r] + 64 + chunk * 4);
        }

        const qs_l0 = ql_vecs[chunk * 2];
        const qs_h0_raw = qh_vecs[chunk * 2];
        const qs_h0 = if (sb_hi) (qs_h0_raw >> shift2) else qs_h0_raw;
        const q6_l0: V = @bitCast(repack8.sli4U8x16(qs_l0, qs_h0 & mask_lo));
        const q6_h0: V = @bitCast((qs_l0 >> shift4) | (qs_h0 & mask_hi));
        inline for (0..M) |r| {
            sb_acc_l0[r] = act_mod.sdotAcc(sb_acc_l0[r], q6_l0, q8_l[r]);
            sb_acc_h0[r] = act_mod.sdotAcc(sb_acc_h0[r], q6_h0, q8_h[r]);
        }

        const qs_l1 = ql_vecs[chunk * 2 + 1];
        const qs_h1_raw = qh_vecs[chunk * 2 + 1];
        const qs_h1 = if (sb_hi) (qs_h1_raw >> shift2) else qs_h1_raw;
        const q6_l1: V = @bitCast(repack8.sli4U8x16(qs_l1, qs_h1 & mask_lo));
        const q6_h1: V = @bitCast((qs_l1 >> shift4) | (qs_h1 & mask_hi));
        inline for (0..M) |r| {
            sb_acc_l1[r] = act_mod.sdotAcc(sb_acc_l1[r], q6_l1, q8_l[r]);
            sb_acc_h1[r] = act_mod.sdotAcc(sb_acc_h1[r], q6_h1, q8_h[r]);
        }
    }

    const sc_l = repack8.loadScalePair(scales_l);
    const sc_h = repack8.loadScalePair(scales_h);
    inline for (0..M) |r| {
        acc[r][0] +%= sb_acc_l0[r] *% @as(I, sc_l.lo) +% sb_acc_h0[r] *% @as(I, sc_h.lo);
        acc[r][1] +%= sb_acc_l1[r] *% @as(I, sc_l.hi) +% sb_acc_h1[r] *% @as(I, sc_h.hi);
    }
}

/// Multi-row Q6_K_8x4 block dot: unpacks 6-bit weights once and reuses them across M prompt rows.
pub inline fn dotQ6Kx8BlockN(
    comptime M: usize,
    blk: *const repack8.BlockQ6Kx8,
    acts: [M]*const sdot.Activation256,
    acc_f32: *[M][2]F4,
) void {
    const q6_d_0: F4 = @floatCast(@as(@Vector(4, f16), blk.d[0..4].*));
    const q6_d_1: F4 = @floatCast(@as(@Vector(4, f16), blk.d[4..8].*));

    var bias_lo: [M]I = @splat(@splat(0));
    var bias_hi: [M]I = @splat(@splat(0));
    const scales_ptr: [*]const i8 = &blk.scales;
    for (0..16) |i| {
        const sc = repack8.loadScalePair(scales_ptr + i * 8);
        inline for (0..M) |r| {
            const bsum: I = @splat(@as(i32, acts[r].bsums[i]));
            bias_lo[r] +%= @as(I, sc.lo) *% bsum;
            bias_hi[r] +%= @as(I, sc.hi) *% bsum;
        }
    }
    inline for (0..M) |r| {
        bias_lo[r] <<= @splat(5);
        bias_hi[r] <<= @splat(5);
    }

    var acc: [M][2]I = @splat(.{ @splat(0), @splat(0) });
    for (0..2) |h| {
        const ql_base: [*]const @Vector(16, u8) = @ptrCast(@alignCast(@as([*]const u8, &blk.ql) + h * 512));
        const qh_base: [*]const @Vector(16, u8) = @ptrCast(@alignCast(@as([*]const u8, &blk.qh) + h * 256));
        const sh = scales_ptr + h * 64;

        for (0..2) |sb| {
            var xh_l: [M][*]const i8 = undefined;
            inline for (0..M) |r| xh_l[r] = @as([*]const i8, &acts[r].values) + h * 128 + sb * 16;
            dotQ6Kx8SubblockN(M, false, ql_base + sb * 8, qh_base + sb * 8, xh_l, sh + sb * 8, sh + (sb + 4) * 8, &acc);
        }
        for (0..2) |sb| {
            var xh_l: [M][*]const i8 = undefined;
            inline for (0..M) |r| xh_l[r] = @as([*]const i8, &acts[r].values) + h * 128 + (sb + 2) * 16;
            dotQ6Kx8SubblockN(M, true, ql_base + (sb + 2) * 8, qh_base + sb * 8, xh_l, sh + (sb + 2) * 8, sh + (sb + 6) * 8, &acc);
        }
    }

    inline for (0..M) |r| {
        const q8_d: F4 = @splat(acts[r].scale);
        const sb_scale_0 = q6_d_0 * q8_d;
        const sb_scale_1 = q6_d_1 * q8_d;
        acc[r][0] -%= bias_lo[r];
        acc[r][1] -%= bias_hi[r];
        acc_f32[r][0] += @as(F4, @floatFromInt(acc[r][0])) * sb_scale_0;
        acc_f32[r][1] += @as(F4, @floatFromInt(acc[r][1])) * sb_scale_1;
    }
}

fn fillQ4K(a: std.mem.Allocator, width: usize, n: usize, seed: u64) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const bpr = width / 256;
    const raw = try a.alloc(u8, n * bpr * 144);
    for (0..n * bpr) |bi| {
        const b = raw[bi * 144 ..][0..144];
        const d: f16 = @floatCast(0.01 + rng.float(f32) * 0.25);
        const m: f16 = @floatCast(0.005 + rng.float(f32) * 0.15);
        std.mem.writeInt(u16, b[0..2], @bitCast(d), .little);
        std.mem.writeInt(u16, b[2..4], @bitCast(m), .little);
        for (b[4..]) |*byte| byte.* = rng.int(u8);
    }
    return raw;
}

test "multi-row Q4_K_8x4 matches 1-row byte for byte across counts, rounding, and thread counts" {
    if (!act_mod.has_dotprod) return error.SkipZigTest;
    const a = std.testing.allocator;
    const width: usize = 512;
    const n: usize = 27;
    const raw = try fillQ4K(a, width, n, 0x9e3779b97f4a7c15);
    defer a.free(raw);
    const t = g.Tensor{ .name = "q4_batch", .kind = .q4_k, .rank = 2, .shape = .{ width, n, 1, 1 }, .offset = 0, .len = raw.len, .bytes = raw };
    const rep = (try repack8.repack(a, t)).?;
    defer a.free(rep.q4_k);

    var prng = std.Random.DefaultPrng.init(0x517cc1b727220a95);
    const rng = prng.random();
    const max_rows: usize = 8;
    const x_all = try a.alloc(f32, max_rows * width);
    defer a.free(x_all);
    for (x_all) |*v| v.* = (rng.float(f32) - 0.5) * 4.0;

    for ([_]bool{ false, true }) |rounded| {
        const ref = try a.alloc(f32, max_rows * n);
        defer a.free(ref);
        for (0..max_rows) |r| {
            try repack8.projectBatch(a, rep, t, x_all[r * width ..][0..width], ref[r * n ..][0..n], 1, rounded);
        }

        for ([_]usize{ 1, 2, 3, 4, 5, 7, 8 }) |count| {
            const got = try a.alloc(f32, count * n);
            defer a.free(got);
            try repack8.projectBatch(a, rep, t, x_all[0 .. count * width], got, count, rounded);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ref[0 .. count * n]), std.mem.sliceAsBytes(got));

            for ([_]usize{ 2, 3, 4, 8 }) |threads| {
                const pool = try Pool.create(a, std.testing.io, threads);
                defer pool.destroy();
                @memset(got, 0);
                try pool.projectSdotRepacked(a, rep, t, x_all[0 .. count * width], got, count, rounded);
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ref[0 .. count * n]), std.mem.sliceAsBytes(got));
            }
        }
    }
}

fn fillQ6K(a: std.mem.Allocator, width: usize, n: usize, seed: u64) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const bpr = width / 256;
    const raw = try a.alloc(u8, n * bpr * 210);
    for (0..n * bpr) |bi| {
        const b = raw[bi * 210 ..][0..210];
        for (b[0..208]) |*byte| byte.* = rng.int(u8);
        const d: f16 = @floatCast(0.01 + rng.float(f32) * 0.25);
        std.mem.writeInt(u16, b[208..210], @bitCast(d), .little);
    }
    return raw;
}

test "multi-row Q6_K_8x4 matches 1-row byte for byte across counts, rounding, and thread counts" {
    if (!act_mod.has_dotprod) return error.SkipZigTest;
    const a = std.testing.allocator;
    const width: usize = 512;
    const n: usize = 27;
    const raw = try fillQ6K(a, width, n, 0xd1b54a32d192ed03);
    defer a.free(raw);
    const t = g.Tensor{ .name = "q6_batch", .kind = .q6_k, .rank = 2, .shape = .{ width, n, 1, 1 }, .offset = 0, .len = raw.len, .bytes = raw };
    const rep = (try repack8.repack(a, t)).?;
    defer a.free(rep.q6_k);

    var prng = std.Random.DefaultPrng.init(0x94d049bb133111eb);
    const rng = prng.random();
    const max_rows: usize = 8;
    const x_all = try a.alloc(f32, max_rows * width);
    defer a.free(x_all);
    for (x_all) |*v| v.* = (rng.float(f32) - 0.5) * 4.0;

    for ([_]bool{ false, true }) |rounded| {
        const ref = try a.alloc(f32, max_rows * n);
        defer a.free(ref);
        for (0..max_rows) |r| {
            try repack8.projectBatch(a, rep, t, x_all[r * width ..][0..width], ref[r * n ..][0..n], 1, rounded);
        }

        for ([_]usize{ 1, 2, 3, 4, 5, 7, 8 }) |count| {
            const got = try a.alloc(f32, count * n);
            defer a.free(got);
            try repack8.projectBatch(a, rep, t, x_all[0 .. count * width], got, count, rounded);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ref[0 .. count * n]), std.mem.sliceAsBytes(got));

            for ([_]usize{ 2, 3, 4, 8 }) |threads| {
                const pool = try Pool.create(a, std.testing.io, threads);
                defer pool.destroy();
                @memset(got, 0);
                try pool.projectSdotRepacked(a, rep, t, x_all[0 .. count * width], got, count, rounded);
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ref[0 .. count * n]), std.mem.sliceAsBytes(got));
            }
        }
    }
}
