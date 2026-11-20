//! 8x4 Q4_K/Q6_K SDOT layout adapted from llama.cpp (MIT, Copyright (c) 2023-2026 The ggml authors).
const std = @import("std");
const core = @import("core");
const g = core.gguf;
const rows = @import("cpu_rows.zig");
const sdot = @import("cpu_gguf_q4_sdot.zig");
const act_mod = @import("cpu_gguf_sdot_act.zig");
const batch_mod = @import("cpu_gguf_repack8_batch.zig");

const V = act_mod.V;
const I = act_mod.I;
const F4 = @Vector(4, f32);
const sdotAcc = act_mod.sdotAcc;

pub inline fn sdotLane(acc: I, a: V, b: V, comptime lane: u2) I {
    if (comptime act_mod.has_dotprod) {
        const idx = switch (lane) {
            0 => "0",
            1 => "1",
            2 => "2",
            3 => "3",
        };
        var out = acc;
        asm ("sdot %[out].4s, %[a].16b, %[b].4b[" ++ idx ++ "]"
            : [out] "+w" (out),
            : [a] "w" (a),
              [b] "x" (b),
        );
        return out;
    }
    const words: @Vector(4, u32) = @bitCast(b);
    const dup: V = @bitCast(@as(@Vector(4, u32), @splat(words[lane])));
    return sdotAcc(acc, a, dup);
}

pub inline fn loadU8Pair(p: [*]const u8) struct { lo: @Vector(4, u16), hi: @Vector(4, u16) } {
    const raw: @Vector(8, u8) = @as(*align(1) const [8]u8, @ptrCast(p)).*;
    const wide: @Vector(8, u16) = raw;
    return .{
        .lo = @shuffle(u16, wide, undefined, [4]i32{ 0, 1, 2, 3 }),
        .hi = @shuffle(u16, wide, undefined, [4]i32{ 4, 5, 6, 7 }),
    };
}

pub inline fn sli4U8x16(dst: @Vector(16, u8), src: @Vector(16, u8)) @Vector(16, u8) {
    if (@import("builtin").cpu.arch == .aarch64) {
        var out = dst;
        asm ("sli %[out].16b, %[src].16b, #4"
            : [out] "+w" (out),
            : [src] "w" (src),
        );
        return out;
    }
    return (dst & @as(@Vector(16, u8), @splat(0x0F))) | (src << @as(@Vector(16, u8), @splat(4)));
}

pub inline fn loadDupI8x4(p: [*]const i8) V {
    const word = std.mem.readInt(u32, @as([*]const u8, @ptrCast(p))[0..4], .little);
    return @bitCast(@as(@Vector(4, u32), @splat(word)));
}

pub inline fn loadScalePair(p: [*]const i8) struct { lo: @Vector(4, i16), hi: @Vector(4, i16) } {
    const raw: @Vector(8, i8) = @as(*align(1) const [8]i8, @ptrCast(p)).*;
    const wide: @Vector(8, i16) = raw;
    return .{
        .lo = @shuffle(i16, wide, undefined, [4]i32{ 0, 1, 2, 3 }),
        .hi = @shuffle(i16, wide, undefined, [4]i32{ 4, 5, 6, 7 }),
    };
}

inline fn dotQ6Kx8Subblock(
    comptime sb_hi: bool,
    ql_vecs: [*]const @Vector(16, u8),
    qh_vecs: [*]const @Vector(16, u8),
    q8_base_l: [*]const i8,
    scales_l: [*]const i8,
    scales_h: [*]const i8,
    acc: *[2]I,
) void {
    const mask_lo: @Vector(16, u8) = @splat(0x03);
    const mask_hi: @Vector(16, u8) = @splat(0x30);
    const shift2: @Vector(16, u8) = @splat(2);
    const shift4: @Vector(16, u8) = @splat(4);
    const q8_base_h = q8_base_l + 64;

    var sb_acc_l0: I = @splat(0);
    var sb_acc_h0: I = @splat(0);
    var sb_acc_l1: I = @splat(0);
    var sb_acc_h1: I = @splat(0);

    inline for (0..4) |chunk| {
        const q8_l = loadDupI8x4(q8_base_l + chunk * 4);
        const q8_h = loadDupI8x4(q8_base_h + chunk * 4);

        const qs_l0 = ql_vecs[chunk * 2];
        const qs_h0_raw = qh_vecs[chunk * 2];
        const qs_h0 = if (sb_hi) (qs_h0_raw >> shift2) else qs_h0_raw;
        const q6_l0: V = @bitCast(sli4U8x16(qs_l0, qs_h0 & mask_lo));
        const q6_h0: V = @bitCast((qs_l0 >> shift4) | (qs_h0 & mask_hi));
        sb_acc_l0 = sdotAcc(sb_acc_l0, q6_l0, q8_l);
        sb_acc_h0 = sdotAcc(sb_acc_h0, q6_h0, q8_h);

        const qs_l1 = ql_vecs[chunk * 2 + 1];
        const qs_h1_raw = qh_vecs[chunk * 2 + 1];
        const qs_h1 = if (sb_hi) (qs_h1_raw >> shift2) else qs_h1_raw;
        const q6_l1: V = @bitCast(sli4U8x16(qs_l1, qs_h1 & mask_lo));
        const q6_h1: V = @bitCast((qs_l1 >> shift4) | (qs_h1 & mask_hi));
        sb_acc_l1 = sdotAcc(sb_acc_l1, q6_l1, q8_l);
        sb_acc_h1 = sdotAcc(sb_acc_h1, q6_h1, q8_h);
    }

    const sc_l = loadScalePair(scales_l);
    const sc_h = loadScalePair(scales_h);
    acc[0] +%= sb_acc_l0 *% @as(I, sc_l.lo) +% sb_acc_h0 *% @as(I, sc_h.lo);
    acc[1] +%= sb_acc_l1 *% @as(I, sc_l.hi) +% sb_acc_h1 *% @as(I, sc_h.hi);
}

inline fn dotQ6Kx8Block(blk: *const BlockQ6Kx8, act: *const sdot.Activation256, acc_f32: *[2]F4) void {
    const q6_d_0: F4 = @floatCast(@as(@Vector(4, f16), blk.d[0..4].*));
    const q6_d_1: F4 = @floatCast(@as(@Vector(4, f16), blk.d[4..8].*));
    const q8_d: F4 = @splat(act.scale);
    const sb_scale_0 = q6_d_0 * q8_d;
    const sb_scale_1 = q6_d_1 * q8_d;

    var bias_lo: I = @splat(0);
    var bias_hi: I = @splat(0);
    const scales_ptr: [*]const i8 = &blk.scales;
    for (0..16) |i| {
        const sc = loadScalePair(scales_ptr + i * 8);
        const bsum: I = @splat(@as(i32, act.bsums[i]));
        bias_lo +%= @as(I, sc.lo) *% bsum;
        bias_hi +%= @as(I, sc.hi) *% bsum;
    }
    bias_lo <<= @splat(5);
    bias_hi <<= @splat(5);

    var acc = [2]I{ @splat(0), @splat(0) };
    const xv: [*]const i8 = &act.values;

    for (0..2) |h| {
        const ql_base: [*]const @Vector(16, u8) = @ptrCast(@alignCast(@as([*]const u8, &blk.ql) + h * 512));
        const qh_base: [*]const @Vector(16, u8) = @ptrCast(@alignCast(@as([*]const u8, &blk.qh) + h * 256));
        const xh = xv + h * 128;
        const sh = scales_ptr + h * 64;

        for (0..2) |sb| {
            dotQ6Kx8Subblock(false, ql_base + sb * 8, qh_base + sb * 8, xh + sb * 16, sh + sb * 8, sh + (sb + 4) * 8, &acc);
        }
        for (0..2) |sb| {
            dotQ6Kx8Subblock(true, ql_base + (sb + 2) * 8, qh_base + sb * 8, xh + (sb + 2) * 16, sh + (sb + 2) * 8, sh + (sb + 6) * 8, &acc);
        }
    }

    acc[0] -%= bias_lo;
    acc[1] -%= bias_hi;
    acc_f32[0] += @as(F4, @floatFromInt(acc[0])) * sb_scale_0;
    acc_f32[1] += @as(F4, @floatFromInt(acc[1])) * sb_scale_1;
}

/// 8 output rows of one 256-value Q4_K block interleaved in 4-byte SDOT lanes (1184 bytes = 148 B/row, +2.8% vs 144 B).
pub const BlockQ4Kx8 = extern struct {
    d: [8]f16 align(16),
    m: [8]f16 align(16),
    sc: [8][8]u8 align(16),
    mn: [8][8]u8 align(16),
    qs: [1024]u8 align(16),
};

/// 8 output rows of one Q6_K block in 4-byte SDOT lanes (1680 B = exact 210 B/row).
pub const BlockQ6Kx8 = extern struct {
    d: [8]f16 align(16),
    ql: [1024]u8 align(16),
    qh: [512]u8 align(16),
    scales: [128]i8 align(16),
};

pub const Repacked = union(enum) {
    q4_k: []const BlockQ4Kx8,
    q6_k: []const BlockQ6Kx8,

    pub fn deinit(self: Repacked, a: std.mem.Allocator) void {
        switch (self) {
            .q4_k => |buf| a.free(buf),
            .q6_k => |buf| a.free(buf),
        }
    }
};

pub fn releaseSourcePages(bytes: []const u8) void {
    const page: usize = std.heap.page_size_min;
    const start = @intFromPtr(bytes.ptr);
    const end = start + bytes.len;
    const aligned_start = (start + page - 1) & ~(page - 1);
    const aligned_end = end & ~(page - 1);
    if (aligned_end <= aligned_start) return;
    const ptr: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(aligned_start);
    std.posix.madvise(ptr, aligned_end - aligned_start, std.posix.MADV.DONTNEED) catch {};
}

pub fn repackQ4KInto(src_bytes: []const u8, width: usize, n: usize, dst: []BlockQ4Kx8) void {
    const blocks_per_row = width / 256;
    const groups = n / 8;
    const row_bytes = blocks_per_row * 144;
    for (0..groups) |grp| {
        for (0..blocks_per_row) |bi| {
            const out = &dst[grp * blocks_per_row + bi];
            var row_ptrs: [8][*]const u8 = undefined;
            inline for (0..8) |r| {
                const src = src_bytes.ptr + (grp * 8 + r) * row_bytes + bi * 144;
                row_ptrs[r] = src;
                out.d[r] = @bitCast(std.mem.readInt(u16, src[0..2], .little));
                out.m[r] = @bitCast(std.mem.readInt(u16, src[2..4], .little));
                const scales = src + 4;
                for (0..4) |k| {
                    out.sc[k][r] = scales[k] & 63;
                    out.mn[k][r] = scales[k + 4] & 63;
                }
                for (4..8) |k| {
                    out.sc[k][r] = (scales[k + 4] & 15) | ((scales[k - 4] >> 6) << 4);
                    out.mn[k][r] = (scales[k + 4] >> 4) | ((scales[k] >> 6) << 4);
                }
            }
            for (0..32) |chunk| {
                inline for (0..8) |r| {
                    @memcpy(out.qs[(chunk * 8 + r) * 4 ..][0..4], row_ptrs[r][16 + chunk * 4 ..][0..4]);
                }
            }
        }
    }
}

pub fn repackQ6KInto(src_bytes: []const u8, width: usize, n: usize, dst: []BlockQ6Kx8) void {
    const blocks_per_row = width / 256;
    const groups = n / 8;
    const row_bytes = blocks_per_row * 210;
    for (0..groups) |grp| {
        for (0..blocks_per_row) |bi| {
            const out = &dst[grp * blocks_per_row + bi];
            var row_ptrs: [8][*]const u8 = undefined;
            inline for (0..8) |r| {
                const src = src_bytes.ptr + (grp * 8 + r) * row_bytes + bi * 210;
                row_ptrs[r] = src;
                out.d[r] = @bitCast(std.mem.readInt(u16, src[208..210], .little));
                for (0..16) |k| out.scales[k * 8 + r] = @bitCast(src[192 + k]);
            }
            for (0..32) |chunk| {
                inline for (0..8) |r| {
                    @memcpy(out.ql[(chunk * 8 + r) * 4 ..][0..4], row_ptrs[r][chunk * 4 ..][0..4]);
                }
            }
            for (0..16) |chunk| {
                inline for (0..8) |r| {
                    @memcpy(out.qh[(chunk * 8 + r) * 4 ..][0..4], row_ptrs[r][128 + chunk * 4 ..][0..4]);
                }
            }
        }
    }
}

pub fn repack(a: std.mem.Allocator, t: g.Tensor) !?Repacked {
    if (t.rank != 2) return error.BadShape;
    const width: usize = @intCast(t.shape[0]);
    const n: usize = @intCast(t.shape[1]);
    if (n < 8 or width == 0 or width % 256 != 0) return null;
    const blocks_per_row = width / 256;
    const groups = n / 8;

    switch (t.kind) {
        .q4_k => {
            const row_bytes = blocks_per_row * 144;
            if (t.bytes.len != n * row_bytes) return error.BadShape;
            const out = try a.alloc(BlockQ4Kx8, groups * blocks_per_row);
            repackQ4KInto(t.bytes, width, n, out);
            return .{ .q4_k = out };
        },
        .q6_k => {
            const row_bytes = blocks_per_row * 210;
            if (t.bytes.len != n * row_bytes) return error.BadShape;
            const out = try a.alloc(BlockQ6Kx8, groups * blocks_per_row);
            repackQ6KInto(t.bytes, width, n, out);
            return .{ .q6_k = out };
        },
        else => return null,
    }
}

inline fn dotQ4Kx8Block(blk: *const BlockQ4Kx8, act: *const sdot.Activation256, acc_f32: *[2]F4) void {
    const mask4: @Vector(16, u8) = @splat(0x0F);
    const shift4: @Vector(16, u8) = @splat(4);

    const q4_d_0: F4 = @floatCast(@as(@Vector(4, f16), blk.d[0..4].*));
    const q4_d_1: F4 = @floatCast(@as(@Vector(4, f16), blk.d[4..8].*));
    const q4_m_0: F4 = @floatCast(@as(@Vector(4, f16), blk.m[0..4].*));
    const q4_m_1: F4 = @floatCast(@as(@Vector(4, f16), blk.m[4..8].*));
    const q8_d: F4 = @splat(act.scale);
    const sb_scale_0 = q4_d_0 * q8_d;
    const sb_scale_1 = q4_d_1 * q8_d;
    const sb_min_0 = q4_m_0 * q8_d;
    const sb_min_1 = q4_m_1 * q8_d;

    var sum_acc = [2]I{ @splat(0), @splat(0) };
    var bias_acc = [2]I{ @splat(0), @splat(0) };
    const x_vecs: [*]const V = @ptrCast(@alignCast(&act.values));
    const qs_vecs: [*]const @Vector(16, u8) = @ptrCast(@alignCast(&blk.qs));

    for (0..4) |sb| {
        const q8_0 = x_vecs[sb * 4];
        const q8_1 = x_vecs[sb * 4 + 1];
        const q8_2 = x_vecs[sb * 4 + 2];
        const q8_3 = x_vecs[sb * 4 + 3];

        const q4_base = qs_vecs + sb * 16;
        const sc_lo = loadU8Pair(&blk.sc[2 * sb]);
        const sc_hi = loadU8Pair(&blk.sc[2 * sb + 1]);

        inline for (0..2) |cg| {
            var acc_lo: I = @splat(0);
            var acc_hi: I = @splat(0);
            inline for (0..4) |i| {
                const raw = q4_base[i * 2 + cg];
                acc_lo = sdotLane(acc_lo, @bitCast(raw & mask4), q8_0, @intCast(i));
                acc_hi = sdotLane(acc_hi, @bitCast(raw >> shift4), q8_2, @intCast(i));
            }
            inline for (0..4) |i| {
                const raw = q4_base[(i + 4) * 2 + cg];
                acc_lo = sdotLane(acc_lo, @bitCast(raw & mask4), q8_1, @intCast(i));
                acc_hi = sdotLane(acc_hi, @bitCast(raw >> shift4), q8_3, @intCast(i));
            }
            const s_lo: I = if (cg == 0) sc_lo.lo else sc_lo.hi;
            const s_hi: I = if (cg == 0) sc_hi.lo else sc_hi.hi;
            sum_acc[cg] +%= acc_lo *% s_lo +% acc_hi *% s_hi;
        }

        const mn_lo = loadU8Pair(&blk.mn[2 * sb]);
        const mn_hi = loadU8Pair(&blk.mn[2 * sb + 1]);
        const sum_lo: I = @splat(act.sums[2 * sb]);
        const sum_hi: I = @splat(act.sums[2 * sb + 1]);
        bias_acc[0] +%= @as(I, mn_lo.lo) *% sum_lo +% @as(I, mn_hi.lo) *% sum_hi;
        bias_acc[1] +%= @as(I, mn_lo.hi) *% sum_lo +% @as(I, mn_hi.hi) *% sum_hi;
    }

    acc_f32[0] += sb_scale_0 * @as(F4, @floatFromInt(sum_acc[0])) - sb_min_0 * @as(F4, @floatFromInt(bias_acc[0]));
    acc_f32[1] += sb_scale_1 * @as(F4, @floatFromInt(sum_acc[1])) - sb_min_1 * @as(F4, @floatFromInt(bias_acc[1]));
}

/// 8-row Q4_K SDOT kernel over 4-byte-interleaved blocks; handles tail rows when hi_grp == n/8.
pub fn projectRangeQ4Kx8(
    blocks: [*]const BlockQ4Kx8,
    raw_bytes: [*]const u8,
    acts: [*]const sdot.Activation256,
    output: [*]f32,
    count: usize,
    blocks_per_row: usize,
    n: usize,
    rounded: bool,
    lo_grp: usize,
    hi_grp: usize,
) void {
    const total_groups = n / 8;

    if (count == 1) {
        var grp = lo_grp;
        while (grp < hi_grp) : (grp +%= 1) {
            const base_blk = blocks + grp *% blocks_per_row;
            var acc_f32 = [2]F4{ @splat(0), @splat(0) };
            var bi: usize = 0;
            while (bi < blocks_per_row) : (bi +%= 1) {
                dotQ4Kx8Block(&base_blk[bi], &acts[bi], &acc_f32);
            }
            const out_base = grp *% 8;
            const lo_arr: [4]f32 = acc_f32[0];
            const hi_arr: [4]f32 = acc_f32[1];
            inline for (0..4) |r| {
                output[out_base +% r] = if (rounded) rows.round(lo_arr[r]) else lo_arr[r];
                output[out_base +% 4 +% r] = if (rounded) rows.round(hi_arr[r]) else hi_arr[r];
            }
        }
        if (hi_grp == total_groups and total_groups *% 8 < n) {
            sdot.projectRangeQ4K(raw_bytes, acts, output, 1, blocks_per_row, n, rounded, total_groups *% 8, n);
        }
        return;
    }

    var grp = lo_grp;
    while (grp < hi_grp) : (grp +%= 1) {
        const base_blk = blocks + grp *% blocks_per_row;
        var first: usize = 0;
        while (first < count) : (first +%= 8) {
            const batch = @min(8, count - first);
            var totals: [8][2]F4 = @splat(.{ @splat(0), @splat(0) });
            var bi: usize = 0;
            while (bi < blocks_per_row) : (bi +%= 1) {
                const blk = &base_blk[bi];
                var b_row: usize = 0;
                while (b_row + 4 <= batch) : (b_row +%= 4) {
                    const row_acts = [4]*const sdot.Activation256{
                        &acts[(first +% b_row) *% blocks_per_row +% bi],
                        &acts[(first +% b_row +% 1) *% blocks_per_row +% bi],
                        &acts[(first +% b_row +% 2) *% blocks_per_row +% bi],
                        &acts[(first +% b_row +% 3) *% blocks_per_row +% bi],
                    };
                    batch_mod.dotQ4Kx8BlockN(4, blk, row_acts, totals[b_row..][0..4]);
                }
                while (b_row + 2 <= batch) : (b_row +%= 2) {
                    const row_acts = [2]*const sdot.Activation256{
                        &acts[(first +% b_row) *% blocks_per_row +% bi],
                        &acts[(first +% b_row +% 1) *% blocks_per_row +% bi],
                    };
                    batch_mod.dotQ4Kx8BlockN(2, blk, row_acts, totals[b_row..][0..2]);
                }
                while (b_row < batch) : (b_row +%= 1) {
                    dotQ4Kx8Block(blk, &acts[(first +% b_row) *% blocks_per_row +% bi], &totals[b_row]);
                }
            }
            const out_col = grp *% 8;
            for (0..batch) |b_row| {
                const row_off = (first +% b_row) *% n +% out_col;
                const lo_arr: [4]f32 = totals[b_row][0];
                const hi_arr: [4]f32 = totals[b_row][1];
                inline for (0..4) |r| {
                    output[row_off +% r] = if (rounded) rows.round(lo_arr[r]) else lo_arr[r];
                    output[row_off +% 4 +% r] = if (rounded) rows.round(hi_arr[r]) else hi_arr[r];
                }
            }
        }
    }
    if (hi_grp == total_groups and total_groups *% 8 < n) {
        sdot.projectRangeQ4K(raw_bytes, acts, output, count, blocks_per_row, n, rounded, total_groups *% 8, n);
    }
}

/// 8-row Q6_K SDOT kernel over 4-byte-interleaved blocks; handles tail rows when hi_grp == n/8.
pub fn projectRangeQ6Kx8(
    blocks: [*]const BlockQ6Kx8,
    raw_bytes: [*]const u8,
    acts: [*]const sdot.Activation256,
    output: [*]f32,
    count: usize,
    blocks_per_row: usize,
    n: usize,
    rounded: bool,
    lo_grp: usize,
    hi_grp: usize,
) void {
    const total_groups = n / 8;

    if (count == 1) {
        var grp = lo_grp;
        while (grp < hi_grp) : (grp +%= 1) {
            const base_blk = blocks + grp *% blocks_per_row;
            var acc_f32 = [2]F4{ @splat(0), @splat(0) };
            var bi: usize = 0;
            while (bi < blocks_per_row) : (bi +%= 1) {
                dotQ6Kx8Block(&base_blk[bi], &acts[bi], &acc_f32);
            }
            const out_base = grp *% 8;
            const lo_arr: [4]f32 = acc_f32[0];
            const hi_arr: [4]f32 = acc_f32[1];
            inline for (0..4) |r| {
                output[out_base +% r] = if (rounded) rows.round(lo_arr[r]) else lo_arr[r];
                output[out_base +% 4 +% r] = if (rounded) rows.round(hi_arr[r]) else hi_arr[r];
            }
        }
        if (hi_grp == total_groups and total_groups *% 8 < n) {
            sdot.projectRangeQ6K(raw_bytes, acts, output, 1, blocks_per_row, n, rounded, total_groups *% 8, n);
        }
        return;
    }

    var grp = lo_grp;
    while (grp < hi_grp) : (grp +%= 1) {
        const base_blk = blocks + grp *% blocks_per_row;
        var first: usize = 0;
        while (first < count) : (first +%= 8) {
            const batch = @min(8, count - first);
            var totals: [8][2]F4 = @splat(.{ @splat(0), @splat(0) });
            var bi: usize = 0;
            while (bi < blocks_per_row) : (bi +%= 1) {
                const blk = &base_blk[bi];
                var b_row: usize = 0;
                while (b_row + 4 <= batch) : (b_row +%= 4) {
                    const row_acts = [4]*const sdot.Activation256{
                        &acts[(first +% b_row) *% blocks_per_row +% bi],
                        &acts[(first +% b_row +% 1) *% blocks_per_row +% bi],
                        &acts[(first +% b_row +% 2) *% blocks_per_row +% bi],
                        &acts[(first +% b_row +% 3) *% blocks_per_row +% bi],
                    };
                    batch_mod.dotQ6Kx8BlockN(4, blk, row_acts, totals[b_row..][0..4]);
                }
                while (b_row + 2 <= batch) : (b_row +%= 2) {
                    const row_acts = [2]*const sdot.Activation256{
                        &acts[(first +% b_row) *% blocks_per_row +% bi],
                        &acts[(first +% b_row +% 1) *% blocks_per_row +% bi],
                    };
                    batch_mod.dotQ6Kx8BlockN(2, blk, row_acts, totals[b_row..][0..2]);
                }
                while (b_row < batch) : (b_row +%= 1) {
                    dotQ6Kx8Block(blk, &acts[(first +% b_row) *% blocks_per_row +% bi], &totals[b_row]);
                }
            }
            const out_col = grp *% 8;
            for (0..batch) |b_row| {
                const row_off = (first +% b_row) *% n +% out_col;
                const lo_arr: [4]f32 = totals[b_row][0];
                const hi_arr: [4]f32 = totals[b_row][1];
                inline for (0..4) |r| {
                    output[row_off +% r] = if (rounded) rows.round(lo_arr[r]) else lo_arr[r];
                    output[row_off +% 4 +% r] = if (rounded) rows.round(hi_arr[r]) else hi_arr[r];
                }
            }
        }
    }
    if (hi_grp == total_groups and total_groups *% 8 < n) {
        sdot.projectRangeQ6K(raw_bytes, acts, output, count, blocks_per_row, n, rounded, total_groups *% 8, n);
    }
}

pub fn projectBatch(a: std.mem.Allocator, repacked: Repacked, t: g.Tensor, input: []const f32, output: []f32, count: usize, rounded: bool) !void {
    try sdot.checkTarget();
    if (t.rank != 2 or count == 0) return error.BadShape;
    const width: usize = @intCast(t.shape[0]);
    const n: usize = @intCast(t.shape[1]);
    if (width == 0 or width % 256 != 0 or n == 0 or input.len != count * width or output.len != count * n) return error.BadShape;
    const blocks_per_row = width / 256;
    const total_acts = count * blocks_per_row;
    if (total_acts <= 8) {
        var small_buf: [8]sdot.Activation256 = undefined;
        const acts = small_buf[0..total_acts];
        for (0..count) |r| {
            try sdot.quantize256(input[r * width ..][0..width], acts[r * blocks_per_row ..][0..blocks_per_row], true);
        }
        switch (repacked) {
            .q4_k => |buf| projectRangeQ4Kx8(buf.ptr, t.bytes.ptr, acts.ptr, output.ptr, count, blocks_per_row, n, rounded, 0, n / 8),
            .q6_k => |buf| projectRangeQ6Kx8(buf.ptr, t.bytes.ptr, acts.ptr, output.ptr, count, blocks_per_row, n, rounded, 0, n / 8),
        }
        return;
    }
    const acts = try a.alloc(sdot.Activation256, total_acts);
    defer a.free(acts);
    for (0..count) |r| {
        try sdot.quantize256(input[r * width ..][0..width], acts[r * blocks_per_row ..][0..blocks_per_row], true);
    }
    switch (repacked) {
        .q4_k => |buf| projectRangeQ4Kx8(buf.ptr, t.bytes.ptr, acts.ptr, output.ptr, count, blocks_per_row, n, rounded, 0, n / 8),
        .q6_k => |buf| projectRangeQ6Kx8(buf.ptr, t.bytes.ptr, acts.ptr, output.ptr, count, blocks_per_row, n, rounded, 0, n / 8),
    }
}

fn expectRepackBitExact(kind: g.TensorKind, width: usize, n: usize, max_count: usize) !void {
    const blk = kind.block();
    const bytes = try std.testing.allocator.alloc(u8, n * (width / 256) * blk.bytes);
    defer std.testing.allocator.free(bytes);
    for (bytes, 0..) |*v, i| v.* = @intCast((i * 73 + 29) % 256);
    for (0..bytes.len / blk.bytes) |b| {
        const part = bytes[b * blk.bytes ..][0..blk.bytes];
        const at: usize = if (kind == .q6_k) 208 else 0;
        std.mem.writeInt(u16, part[at..][0..2], if (b % 3 == 0) 0x3500 else if (b % 3 == 1) 0xb400 else 0x3200, .little);
        if (kind == .q4_k) std.mem.writeInt(u16, part[2..4], if (b % 2 == 0) 0x3100 else 0x3000, .little);
    }
    const t = g.Tensor{ .name = "repack8", .kind = kind, .rank = 2, .shape = .{ width, n, 1, 1 }, .offset = 0, .len = bytes.len, .bytes = bytes };
    const rep = (try repack(std.testing.allocator, t)).?;
    defer rep.deinit(std.testing.allocator);
    const x = try std.testing.allocator.alloc(f32, max_count * width);
    defer std.testing.allocator.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 19) % 41)) - 20)) / 9.0;
    const want = try std.testing.allocator.alloc(f32, max_count * n);
    defer std.testing.allocator.free(want);
    const got = try std.testing.allocator.alloc(f32, max_count * n);
    defer std.testing.allocator.free(got);
    for ([_]bool{ false, true }) |rounded| for ([_]usize{ 1, max_count }) |count| {
        try sdot.projectBatch(std.testing.allocator, t, x[0 .. count * width], want[0 .. count * n], count, rounded);
        try projectBatch(std.testing.allocator, rep, t, x[0 .. count * width], got[0 .. count * n], count, rounded);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want[0 .. count * n]), std.mem.sliceAsBytes(got[0 .. count * n]));
    };
}

test "8-row repacked Q4_K and Q6_K SDOT match 1-row SDOT bit-for-bit including non-multiple-of-8 tails" {
    for ([_]g.TensorKind{ .q4_k, .q6_k }) |kind| for ([_]usize{ 8, 13, 24 }) |n| try expectRepackBitExact(kind, 512, n, 2);
}

test "Q6_K 8x4 interleaved layout matches 1-row SDOT bit-for-bit on Ling width 1536" {
    try std.testing.expectEqual(@as(usize, 1680), @sizeOf(BlockQ6Kx8));
    try expectRepackBitExact(.q6_k, 1536, 24, 4);
}

test "Q4_K 8x4 interleaved layout matches 1-row SDOT bit-for-bit on Ling expert width 1536" {
    try std.testing.expectEqual(@as(usize, 1184), @sizeOf(BlockQ4Kx8));
    try expectRepackBitExact(.q4_k, 1536, 24, 4);
}
