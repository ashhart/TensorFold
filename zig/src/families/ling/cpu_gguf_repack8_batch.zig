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
