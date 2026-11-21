//! Opt-in approximate SDOT kernels for Q4_K, Q6_K, and Q5_0; exact kernels stay separate.
const std = @import("std");
const core = @import("core");
const g = core.gguf;
const rows = @import("cpu_rows.zig");
const act_mod = @import("cpu_gguf_sdot_act.zig");

const V = act_mod.V;
const I = act_mod.I;
const sdotAcc = act_mod.sdotAcc;

pub const Activation256 = act_mod.Activation256;
pub const Activation32 = act_mod.Activation32;
pub const checkTarget = act_mod.checkTarget;
pub const quantize256 = act_mod.quantize256;
pub const quantize32 = act_mod.quantize32;

inline fn half(p: [*]const u8) f32 {
    return @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, p[0..2], .little))));
}

inline fn loadU8x16(p: [*]const u8) @Vector(16, u8) {
    return @as(*align(1) const [16]u8, @ptrCast(p)).*;
}

inline fn loadI8x16Aligned(p: [*]align(16) const i8) V {
    return @as(*align(16) const [16]i8, @ptrCast(p)).*;
}

inline fn expandBits8(byte: u8) @Vector(8, u8) {
    const bit_mask: @Vector(8, u8) = .{ 1, 2, 4, 8, 16, 32, 64, 128 };
    const bits = @as(@Vector(8, u8), @splat(byte)) & bit_mask;
    return @select(u8, bits != @as(@Vector(8, u8), @splat(0)), @as(@Vector(8, u8), @splat(16)), @as(@Vector(8, u8), @splat(0)));
}

inline fn expandBits16(b0: u8, b1: u8) @Vector(16, u8) {
    const lo = expandBits8(b0);
    const hi = expandBits8(b1);
    return @shuffle(u8, lo, hi, [16]i32{ 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8 });
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

/// Inner Q4_K block dot product; wrapping add and mul are overflow-free (max |weighted| = 30.7M << 2^31).
pub inline fn q4kBlockDot(p: [*]const u8, act: *const Activation256) f32 {
    const mask4: @Vector(16, u8) = @splat(0x0F);
    const shift4: @Vector(16, u8) = @splat(4);
    const scales = p + 4;

    const sc0: i32 = scales[0] & 63;
    const sc1: i32 = scales[1] & 63;
    const sc2: i32 = scales[2] & 63;
    const sc3: i32 = scales[3] & 63;
    const sc4: i32 = (scales[8] & 15) | ((scales[0] >> 6) << 4);
    const sc5: i32 = (scales[9] & 15) | ((scales[1] >> 6) << 4);
    const sc6: i32 = (scales[10] & 15) | ((scales[2] >> 6) << 4);
    const sc7: i32 = (scales[11] & 15) | ((scales[3] >> 6) << 4);

    const mn_lo: I = .{
        @intCast(scales[4] & 63),
        @intCast(scales[5] & 63),
        @intCast(scales[6] & 63),
        @intCast(scales[7] & 63),
    };
    const mn_hi: I = .{
        @intCast((scales[8] >> 4) | ((scales[4] >> 6) << 4)),
        @intCast((scales[9] >> 4) | ((scales[5] >> 6) << 4)),
        @intCast((scales[10] >> 4) | ((scales[6] >> 6) << 4)),
        @intCast((scales[11] >> 4) | ((scales[7] >> 6) << 4)),
    };
    const sums_lo: I = @as(*align(16) const [4]i32, @ptrCast(&act.sums[0])).*;
    const sums_hi: I = @as(*align(16) const [4]i32, @ptrCast(&act.sums[4])).*;
    const minimum: i32 = @reduce(.Add, mn_lo *% sums_lo +% mn_hi *% sums_hi);

    const sc_arr = [8]i32{ sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7 };
    const xv: [*]align(16) const i8 = &act.values;
    var sum_vec: I = @splat(0);

    inline for (0..4) |chunk| {
        const qs = p + 16 + chunk * 32;
        const raw0 = loadU8x16(qs);
        const raw1 = loadU8x16(qs + 16);
        const xc = xv + chunk * 64;

        var acc_lo: I = @splat(0);
        acc_lo = sdotAcc(acc_lo, @bitCast(raw0 & mask4), loadI8x16Aligned(xc));
        acc_lo = sdotAcc(acc_lo, @bitCast(raw1 & mask4), loadI8x16Aligned(xc + 16));
        sum_vec +%= acc_lo *% @as(I, @splat(sc_arr[2 * chunk]));

        var acc_hi: I = @splat(0);
        acc_hi = sdotAcc(acc_hi, @bitCast(raw0 >> shift4), loadI8x16Aligned(xc + 32));
        acc_hi = sdotAcc(acc_hi, @bitCast(raw1 >> shift4), loadI8x16Aligned(xc + 48));
        sum_vec +%= acc_hi *% @as(I, @splat(sc_arr[2 * chunk + 1]));
    }
    const weighted: i32 = @reduce(.Add, sum_vec);
    return half(p) * act.scale * @as(f32, @floatFromInt(weighted)) - half(p + 2) * act.scale * @as(f32, @floatFromInt(minimum));
}

pub fn projectRangeQ4K(bytes: [*]const u8, acts: [*]const Activation256, output: [*]f32, count: usize, blocks_per_row: usize, n: usize, rounded: bool, lo: usize, hi: usize) void {
    const row_bytes = blocks_per_row *% 144;
    if (count == 1) {
        var row = lo;
        while (row +% 3 < hi) : (row +%= 4) {
            const r0 = bytes + row *% row_bytes;
            const r1 = r0 + row_bytes;
            const r2 = r1 + row_bytes;
            const r3 = r2 + row_bytes;
            var total0: f32 = 0;
            var total1: f32 = 0;
            var total2: f32 = 0;
            var total3: f32 = 0;
            var bi: usize = 0;
            while (bi < blocks_per_row) : (bi +%= 1) {
                const off = bi *% 144;
                const act = &acts[bi];
                total0 += q4kBlockDot(r0 + off, act);
                total1 += q4kBlockDot(r1 + off, act);
                total2 += q4kBlockDot(r2 + off, act);
                total3 += q4kBlockDot(r3 + off, act);
            }
            output[row] = if (rounded) rows.round(total0) else total0;
            output[row +% 1] = if (rounded) rows.round(total1) else total1;
            output[row +% 2] = if (rounded) rows.round(total2) else total2;
            output[row +% 3] = if (rounded) rows.round(total3) else total3;
        }
        while (row < hi) : (row +%= 1) {
            const r0 = bytes + row *% row_bytes;
            var total0: f32 = 0;
            var bi: usize = 0;
            while (bi < blocks_per_row) : (bi +%= 1) {
                total0 += q4kBlockDot(r0 + bi *% 144, &acts[bi]);
            }
            output[row] = if (rounded) rows.round(total0) else total0;
        }
        return;
    }

    const mask4: @Vector(16, u8) = @splat(0x0F);
    const shift4: @Vector(16, u8) = @splat(4);
    for (lo..hi) |row| {
        const r_ptr = bytes + row *% row_bytes;
        var first: usize = 0;
        while (first < count) : (first +%= 8) {
            const batch = @min(8, count - first);
            var totals: [8]f32 = @splat(0);
            var bi: usize = 0;
            while (bi < blocks_per_row) : (bi +%= 1) {
                const p = r_ptr + bi *% 144;
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
                var sum_vecs: [8]I = @splat(@splat(0));
                var minimum: [8]i32 = @splat(0);
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
                    for (0..batch) |r| {
                        const act = &acts[(first +% r) *% blocks_per_row +% bi];
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
                for (0..batch) |r| {
                    const scale = acts[(first +% r) *% blocks_per_row +% bi].scale;
                    const weighted: i32 = @reduce(.Add, sum_vecs[r]);
                    totals[r] += d * scale * @as(f32, @floatFromInt(weighted)) - m * scale * @as(f32, @floatFromInt(minimum[r]));
                }
            }
            for (0..batch) |r| {
                output[(first +% r) *% n +% row] = if (rounded) rows.round(totals[r]) else totals[r];
            }
        }
    }
}

/// Inner Q6_K block dot product; wrapping add and mul are overflow-free (max |weighted| = 133.2M << 2^31).
pub inline fn q6kBlockDot(p: [*]const u8, act: *const Activation256) f32 {
    const d = half(p + 208);
    const xv: [*]align(16) const i8 = &act.values;
    var sum_vec: I = @splat(0);
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
        const xh = xv + h * 128;
        inline for (0..4) |quarter| {
            const ql_a = if (quarter % 2 == 0) ql0 else ql2;
            const ql_b = if (quarter % 2 == 0) ql1 else ql3;
            const c_a = q6kQuarter16(ql_a, qh0, quarter);
            const c_b = q6kQuarter16(ql_b, qh1, quarter);
            const xq = xh + quarter * 32;
            const s_a: I = @splat(@as(i32, @as(i8, @bitCast(sc_ptr[quarter * 2]))));
            const s_b: I = @splat(@as(i32, @as(i8, @bitCast(sc_ptr[quarter * 2 + 1]))));
            sum_vec +%= sdotAcc(@splat(0), c_a, loadI8x16Aligned(xq)) *% s_a;
            sum_vec +%= sdotAcc(@splat(0), c_b, loadI8x16Aligned(xq + 16)) *% s_b;
        }
    }
    const weighted: i32 = @reduce(.Add, sum_vec);
    return d * act.scale * @as(f32, @floatFromInt(weighted));
}

pub fn projectRangeQ6K(bytes: [*]const u8, acts: [*]const Activation256, output: [*]f32, count: usize, blocks_per_row: usize, n: usize, rounded: bool, lo: usize, hi: usize) void {
    const row_bytes = blocks_per_row *% 210;
    if (count == 1) {
        var row = lo;
        while (row +% 1 < hi) : (row +%= 2) {
            const r0 = bytes + row *% row_bytes;
            const r1 = r0 + row_bytes;
            var total0: f32 = 0;
            var total1: f32 = 0;
            var bi: usize = 0;
            while (bi < blocks_per_row) : (bi +%= 1) {
                const off = bi *% 210;
                const act = &acts[bi];
                total0 += q6kBlockDot(r0 + off, act);
                total1 += q6kBlockDot(r1 + off, act);
            }
            output[row] = if (rounded) rows.round(total0) else total0;
            output[row +% 1] = if (rounded) rows.round(total1) else total1;
        }
        if (row < hi) {
            const r0 = bytes + row *% row_bytes;
            var total0: f32 = 0;
            var bi: usize = 0;
            while (bi < blocks_per_row) : (bi +%= 1) {
                total0 += q6kBlockDot(r0 + bi *% 210, &acts[bi]);
            }
            output[row] = if (rounded) rows.round(total0) else total0;
        }
        return;
    }

    for (lo..hi) |row| {
        const r_ptr = bytes + row *% row_bytes;
        var first: usize = 0;
        while (first < count) : (first +%= 8) {
            const batch = @min(8, count - first);
            var totals: [8]f32 = @splat(0);
            var bi: usize = 0;
            while (bi < blocks_per_row) : (bi +%= 1) {
                const p = r_ptr + bi *% 210;
                const d = half(p + 208);
                var sum_vecs: [8]I = @splat(@splat(0));
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
                        for (0..batch) |r| {
                            const act = &acts[(first +% r) *% blocks_per_row +% bi];
                            const xq = @as([*]align(16) const i8, &act.values) + h * 128 + quarter * 32;
                            sum_vecs[r] +%= sdotAcc(@splat(0), c_a, loadI8x16Aligned(xq)) *% s_a;
                            sum_vecs[r] +%= sdotAcc(@splat(0), c_b, loadI8x16Aligned(xq + 16)) *% s_b;
                        }
                    }
                }
                for (0..batch) |r| {
                    const scale = acts[(first +% r) *% blocks_per_row +% bi].scale;
                    totals[r] += d * scale * @as(f32, @floatFromInt(@reduce(.Add, sum_vecs[r])));
                }
            }
            for (0..batch) |r| {
                output[(first +% r) *% n +% row] = if (rounded) rows.round(totals[r]) else totals[r];
            }
        }
    }
}

inline fn q50BlockDot(p: [*]const u8, act: *const Activation32) f32 {
    const mask4: @Vector(16, u8) = @splat(0x0F);
    const shift4: @Vector(16, u8) = @splat(4);
    const bias16: V = @splat(16);
    const qs = loadU8x16(p + 6);
    const c0: V = @as(V, @bitCast((qs & mask4) | expandBits16(p[2], p[3]))) -% bias16;
    const c1: V = @as(V, @bitCast((qs >> shift4) | expandBits16(p[4], p[5]))) -% bias16;
    const xv: [*]align(16) const i8 = &act.values;
    var acc: I = @splat(0);
    acc = sdotAcc(acc, c0, loadI8x16Aligned(xv));
    acc = sdotAcc(acc, c1, loadI8x16Aligned(xv + 16));
    return half(p) * act.scale * @as(f32, @floatFromInt(@reduce(.Add, acc)));
}

pub fn projectRangeQ50(bytes: [*]const u8, acts: [*]const Activation32, output: [*]f32, count: usize, blocks_per_row: usize, n: usize, rounded: bool, lo: usize, hi: usize) void {
    const row_bytes = blocks_per_row *% 22;
    if (count == 1) {
        var row = lo;
        while (row +% 3 < hi) : (row +%= 4) {
            const r0 = bytes + row *% row_bytes;
            const r1 = r0 + row_bytes;
            const r2 = r1 + row_bytes;
            const r3 = r2 + row_bytes;
            var total0: f32 = 0;
            var total1: f32 = 0;
            var total2: f32 = 0;
            var total3: f32 = 0;
            var bi: usize = 0;
            while (bi < blocks_per_row) : (bi +%= 1) {
                const off = bi *% 22;
                const act = &acts[bi];
                total0 += q50BlockDot(r0 + off, act);
                total1 += q50BlockDot(r1 + off, act);
                total2 += q50BlockDot(r2 + off, act);
                total3 += q50BlockDot(r3 + off, act);
            }
            output[row] = if (rounded) rows.round(total0) else total0;
            output[row +% 1] = if (rounded) rows.round(total1) else total1;
            output[row +% 2] = if (rounded) rows.round(total2) else total2;
            output[row +% 3] = if (rounded) rows.round(total3) else total3;
        }
        while (row < hi) : (row +%= 1) {
            const r0 = bytes + row *% row_bytes;
            var total0: f32 = 0;
            var bi: usize = 0;
            while (bi < blocks_per_row) : (bi +%= 1) {
                total0 += q50BlockDot(r0 + bi *% 22, &acts[bi]);
            }
            output[row] = if (rounded) rows.round(total0) else total0;
        }
        return;
    }

    const mask4: @Vector(16, u8) = @splat(0x0F);
    const shift4: @Vector(16, u8) = @splat(4);
    const bias16: V = @splat(16);
    for (lo..hi) |row| {
        const r_ptr = bytes + row *% row_bytes;
        var first: usize = 0;
        while (first < count) : (first +%= 8) {
            const batch = @min(8, count - first);
            var totals: [8]f32 = @splat(0);
            var bi: usize = 0;
            while (bi < blocks_per_row) : (bi +%= 1) {
                const p = r_ptr + bi *% 22;
                const d = half(p);
                const qs = loadU8x16(p + 6);
                const c0: V = @as(V, @bitCast((qs & mask4) | expandBits16(p[2], p[3]))) -% bias16;
                const c1: V = @as(V, @bitCast((qs >> shift4) | expandBits16(p[4], p[5]))) -% bias16;
                for (0..batch) |r| {
                    const act = &acts[(first +% r) *% blocks_per_row +% bi];
                    const xv: [*]align(16) const i8 = &act.values;
                    var acc: I = @splat(0);
                    acc = sdotAcc(acc, c0, loadI8x16Aligned(xv));
                    acc = sdotAcc(acc, c1, loadI8x16Aligned(xv + 16));
                    totals[r] += d * act.scale * @as(f32, @floatFromInt(@reduce(.Add, acc)));
                }
            }
            for (0..batch) |r| {
                output[(first +% r) *% n +% row] = if (rounded) rows.round(totals[r]) else totals[r];
            }
        }
    }
}

pub fn projectBatch(a: std.mem.Allocator, t: g.Tensor, input: []const f32, output: []f32, count: usize, rounded: bool) !void {
    try checkTarget();
    if (t.rank != 2 or count == 0) return error.BadShape;
    const width: usize = @intCast(t.shape[0]);
    const n: usize = @intCast(t.shape[1]);
    if (width == 0 or n == 0 or input.len != count * width or output.len != count * n) return error.BadShape;
    switch (t.kind) {
        .q4_k, .q6_k => {
            const bsize: usize = if (t.kind == .q4_k) 144 else 210;
            if (width % 256 != 0 or t.bytes.len != n * (width / 256) * bsize) return error.BadShape;
            const blocks_per_row = width / 256;
            const total_acts = count * blocks_per_row;
            if (total_acts <= 8) {
                var small_buf: [8]Activation256 = undefined;
                const acts = small_buf[0..total_acts];
                for (0..count) |r| {
                    if (t.kind == .q4_k) {
                        try quantize256(input[r * width ..][0..width], acts[r * blocks_per_row ..][0..blocks_per_row], true);
                    } else {
                        try quantize256(input[r * width ..][0..width], acts[r * blocks_per_row ..][0..blocks_per_row], false);
                    }
                }
                if (t.kind == .q4_k) {
                    projectRangeQ4K(t.bytes.ptr, acts.ptr, output.ptr, count, blocks_per_row, n, rounded, 0, n);
                } else {
                    projectRangeQ6K(t.bytes.ptr, acts.ptr, output.ptr, count, blocks_per_row, n, rounded, 0, n);
                }
                return;
            }
            const acts = try a.alloc(Activation256, total_acts);
            defer a.free(acts);
            for (0..count) |r| {
                if (t.kind == .q4_k) {
                    try quantize256(input[r * width ..][0..width], acts[r * blocks_per_row ..][0..blocks_per_row], true);
                } else {
                    try quantize256(input[r * width ..][0..width], acts[r * blocks_per_row ..][0..blocks_per_row], false);
                }
            }
            if (t.kind == .q4_k) {
                projectRangeQ4K(t.bytes.ptr, acts.ptr, output.ptr, count, blocks_per_row, n, rounded, 0, n);
            } else {
                projectRangeQ6K(t.bytes.ptr, acts.ptr, output.ptr, count, blocks_per_row, n, rounded, 0, n);
            }
        },
        .q5_0 => {
            if (width % 32 != 0 or t.bytes.len != n * (width / 32) * 22) return error.BadShape;
            const blocks_per_row = width / 32;
            const total_acts = count * blocks_per_row;
            if (total_acts <= 64) {
                var small_buf: [64]Activation32 = undefined;
                const acts = small_buf[0..total_acts];
                for (0..count) |r| {
                    try quantize32(input[r * width ..][0..width], acts[r * blocks_per_row ..][0..blocks_per_row]);
                }
                projectRangeQ50(t.bytes.ptr, acts.ptr, output.ptr, count, blocks_per_row, n, rounded, 0, n);
                return;
            }
            const acts = try a.alloc(Activation32, total_acts);
            defer a.free(acts);
            for (0..count) |r| {
                try quantize32(input[r * width ..][0..width], acts[r * blocks_per_row ..][0..blocks_per_row]);
            }
            projectRangeQ50(t.bytes.ptr, acts.ptr, output.ptr, count, blocks_per_row, n, rounded, 0, n);
        },
        else => return error.UnsupportedLingGgufQuant,
    }
}

pub fn project(a: std.mem.Allocator, t: g.Tensor, x: []const f32, y: []f32, rounded: bool) !void {
    return projectBatch(a, t, x, y, 1, rounded);
}
