//! Exact Q6_K and Q5_0 rows: each dot keeps the eight-lane FMA order of the scalar reference, bit for bit.
const rows = @import("cpu_rows.zig");
const lanes = @import("cpu_gguf_lanes.zig");
const readF16 = lanes.readF16;
const loadF32x8 = lanes.loadF32x8;
const lowHalf = lanes.lowHalf;
const highHalf = lanes.highHalf;
const lowHalfI8 = lanes.lowHalfI8;
const highHalfI8 = lanes.highHalfI8;
const i8x8ToF32x8 = lanes.i8x8ToF32x8;
const expandBits8 = lanes.expandBits8;

inline fn q6kQuarter16(ql: @Vector(16, u8), qh: @Vector(16, u8), comptime quarter: usize) @Vector(16, i8) {
    const mask4: @Vector(16, u8) = @splat(0x0F);
    const lo = if (quarter < 2) (ql & mask4) else (ql >> @as(@Vector(16, u8), @splat(4)));
    const hi = switch (quarter) {
        0 => (qh & @as(@Vector(16, u8), @splat(0x03))) << @as(@Vector(16, u8), @splat(4)),
        1 => (qh & @as(@Vector(16, u8), @splat(0x0C))) << @as(@Vector(16, u8), @splat(2)),
        2 => qh & @as(@Vector(16, u8), @splat(0x30)),
        3 => (qh & @as(@Vector(16, u8), @splat(0xC0))) >> @as(@Vector(16, u8), @splat(2)),
        else => unreachable,
    };
    return @as(@Vector(16, i8), @bitCast(lo | hi)) - @as(@Vector(16, i8), @splat(32));
}

pub fn projectQ6KRange(bytes: [*]const u8, input: [*]const f32, output: [*]f32, count: usize, width: usize, n: usize, rounded: bool, lo: usize, hi: usize) void {
    const blocks_per_row = width / 256;
    const row_bytes = blocks_per_row * 210;

    if (count == 1) {
        var row = lo;
        while (row + 1 < hi) : (row +%= 2) {
            const r0_ptr = bytes + row * row_bytes;
            const r1_ptr = r0_ptr + row_bytes;
            var sum0: @Vector(8, f32) = @splat(0);
            var sum1: @Vector(8, f32) = @splat(0);
            var b: usize = 0;
            while (b < blocks_per_row) : (b +%= 1) {
                const b0 = r0_ptr + b * 210;
                const b1 = r1_ptr + b * 210;
                const d0 = readF16(b0 + 208);
                const d1 = readF16(b1 + 208);
                const x_blk = input + b * 256;

                inline for (0..2) |half| {
                    const ql0_ptr = b0 + half * 64;
                    const qh0_ptr = b0 + 128 + half * 32;
                    const sc0_ptr = b0 + 192 + half * 8;
                    const ql1_ptr = b1 + half * 64;
                    const qh1_ptr = b1 + 128 + half * 32;
                    const sc1_ptr = b1 + 192 + half * 8;

                    const ql0_0: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql0_ptr)).*;
                    const ql0_1: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql0_ptr + 16)).*;
                    const ql0_2: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql0_ptr + 32)).*;
                    const ql0_3: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql0_ptr + 48)).*;
                    const qh0_0: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qh0_ptr)).*;
                    const qh0_1: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qh0_ptr + 16)).*;

                    const ql1_0: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql1_ptr)).*;
                    const ql1_1: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql1_ptr + 16)).*;
                    const ql1_2: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql1_ptr + 32)).*;
                    const ql1_3: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql1_ptr + 48)).*;
                    const qh1_0: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qh1_ptr)).*;
                    const qh1_1: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qh1_ptr + 16)).*;

                    inline for (0..4) |quarter| {
                        const ql0_a = if (quarter % 2 == 0) ql0_0 else ql0_2;
                        const ql0_b = if (quarter % 2 == 0) ql0_1 else ql0_3;
                        const ql1_a = if (quarter % 2 == 0) ql1_0 else ql1_2;
                        const ql1_b = if (quarter % 2 == 0) ql1_1 else ql1_3;

                        const c0_a = q6kQuarter16(ql0_a, qh0_0, quarter);
                        const c0_b = q6kQuarter16(ql0_b, qh0_1, quarter);
                        const c1_a = q6kQuarter16(ql1_a, qh1_0, quarter);
                        const c1_b = q6kQuarter16(ql1_b, qh1_1, quarter);

                        const s0_a: @Vector(8, f32) = @splat(d0 * @as(f32, @floatFromInt(@as(i8, @bitCast(sc0_ptr[quarter * 2])))));
                        const s0_b: @Vector(8, f32) = @splat(d0 * @as(f32, @floatFromInt(@as(i8, @bitCast(sc0_ptr[quarter * 2 + 1])))));
                        const s1_a: @Vector(8, f32) = @splat(d1 * @as(f32, @floatFromInt(@as(i8, @bitCast(sc1_ptr[quarter * 2])))));
                        const s1_b: @Vector(8, f32) = @splat(d1 * @as(f32, @floatFromInt(@as(i8, @bitCast(sc1_ptr[quarter * 2 + 1])))));

                        const xq = x_blk + half * 128 + quarter * 32;
                        const x0 = loadF32x8(xq);
                        sum0 = @mulAdd(@Vector(8, f32), x0, s0_a * i8x8ToF32x8(lowHalfI8(c0_a)), sum0);
                        sum1 = @mulAdd(@Vector(8, f32), x0, s1_a * i8x8ToF32x8(lowHalfI8(c1_a)), sum1);

                        const x1 = loadF32x8(xq + 8);
                        sum0 = @mulAdd(@Vector(8, f32), x1, s0_a * i8x8ToF32x8(highHalfI8(c0_a)), sum0);
                        sum1 = @mulAdd(@Vector(8, f32), x1, s1_a * i8x8ToF32x8(highHalfI8(c1_a)), sum1);

                        const x2 = loadF32x8(xq + 16);
                        sum0 = @mulAdd(@Vector(8, f32), x2, s0_b * i8x8ToF32x8(lowHalfI8(c0_b)), sum0);
                        sum1 = @mulAdd(@Vector(8, f32), x2, s1_b * i8x8ToF32x8(lowHalfI8(c1_b)), sum1);

                        const x3 = loadF32x8(xq + 24);
                        sum0 = @mulAdd(@Vector(8, f32), x3, s0_b * i8x8ToF32x8(highHalfI8(c0_b)), sum0);
                        sum1 = @mulAdd(@Vector(8, f32), x3, s1_b * i8x8ToF32x8(highHalfI8(c1_b)), sum1);
                    }
                }
            }
            const v0 = @reduce(.Add, sum0);
            const v1 = @reduce(.Add, sum1);
            output[row] = if (rounded) rows.round(v0) else v0;
            output[row + 1] = if (rounded) rows.round(v1) else v1;
        }
        if (row < hi) {
            const r0_ptr = bytes + row * row_bytes;
            var sum0: @Vector(8, f32) = @splat(0);
            var b: usize = 0;
            while (b < blocks_per_row) : (b +%= 1) {
                const b0 = r0_ptr + b * 210;
                const d0 = readF16(b0 + 208);
                const x_blk = input + b * 256;
                inline for (0..2) |half| {
                    const ql0_ptr = b0 + half * 64;
                    const qh0_ptr = b0 + 128 + half * 32;
                    const sc0_ptr = b0 + 192 + half * 8;
                    const ql0_0: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql0_ptr)).*;
                    const ql0_1: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql0_ptr + 16)).*;
                    const ql0_2: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql0_ptr + 32)).*;
                    const ql0_3: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql0_ptr + 48)).*;
                    const qh0_0: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qh0_ptr)).*;
                    const qh0_1: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qh0_ptr + 16)).*;
                    inline for (0..4) |quarter| {
                        const ql0_a = if (quarter % 2 == 0) ql0_0 else ql0_2;
                        const ql0_b = if (quarter % 2 == 0) ql0_1 else ql0_3;
                        const c0_a = q6kQuarter16(ql0_a, qh0_0, quarter);
                        const c0_b = q6kQuarter16(ql0_b, qh0_1, quarter);
                        const s0_a: @Vector(8, f32) = @splat(d0 * @as(f32, @floatFromInt(@as(i8, @bitCast(sc0_ptr[quarter * 2])))));
                        const s0_b: @Vector(8, f32) = @splat(d0 * @as(f32, @floatFromInt(@as(i8, @bitCast(sc0_ptr[quarter * 2 + 1])))));
                        const xq = x_blk + half * 128 + quarter * 32;
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(xq), s0_a * i8x8ToF32x8(lowHalfI8(c0_a)), sum0);
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(xq + 8), s0_a * i8x8ToF32x8(highHalfI8(c0_a)), sum0);
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(xq + 16), s0_b * i8x8ToF32x8(lowHalfI8(c0_b)), sum0);
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(xq + 24), s0_b * i8x8ToF32x8(highHalfI8(c0_b)), sum0);
                    }
                }
            }
            const v0 = @reduce(.Add, sum0);
            output[row] = if (rounded) rows.round(v0) else v0;
        }
        return;
    }

    if (count == 2) {
        const input1 = input + width;
        for (lo..hi) |row| {
            const r_ptr = bytes + row * row_bytes;
            var sum0: @Vector(8, f32) = @splat(0);
            var sum1: @Vector(8, f32) = @splat(0);
            var b: usize = 0;
            while (b < blocks_per_row) : (b +%= 1) {
                const blk = r_ptr + b * 210;
                const d = readF16(blk + 208);
                const x0_blk = input + b * 256;
                const x1_blk = input1 + b * 256;
                inline for (0..2) |half| {
                    const ql_ptr = blk + half * 64;
                    const qh_ptr = blk + 128 + half * 32;
                    const sc_ptr = blk + 192 + half * 8;
                    const ql_0: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql_ptr)).*;
                    const ql_1: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql_ptr + 16)).*;
                    const ql_2: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql_ptr + 32)).*;
                    const ql_3: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql_ptr + 48)).*;
                    const qh_0: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qh_ptr)).*;
                    const qh_1: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qh_ptr + 16)).*;
                    inline for (0..4) |quarter| {
                        const ql_a = if (quarter % 2 == 0) ql_0 else ql_2;
                        const ql_b = if (quarter % 2 == 0) ql_1 else ql_3;
                        const c_a = q6kQuarter16(ql_a, qh_0, quarter);
                        const c_b = q6kQuarter16(ql_b, qh_1, quarter);
                        const s_a: @Vector(8, f32) = @splat(d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc_ptr[quarter * 2])))));
                        const s_b: @Vector(8, f32) = @splat(d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc_ptr[quarter * 2 + 1])))));
                        const x0q = x0_blk + half * 128 + quarter * 32;
                        const x1q = x1_blk + half * 128 + quarter * 32;
                        const w0 = s_a * i8x8ToF32x8(lowHalfI8(c_a));
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(x0q), w0, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), loadF32x8(x1q), w0, sum1);
                        const w1 = s_a * i8x8ToF32x8(highHalfI8(c_a));
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(x0q + 8), w1, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), loadF32x8(x1q + 8), w1, sum1);
                        const w2 = s_b * i8x8ToF32x8(lowHalfI8(c_b));
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(x0q + 16), w2, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), loadF32x8(x1q + 16), w2, sum1);
                        const w3 = s_b * i8x8ToF32x8(highHalfI8(c_b));
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(x0q + 24), w3, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), loadF32x8(x1q + 24), w3, sum1);
                    }
                }
            }
            const v0 = @reduce(.Add, sum0);
            const v1 = @reduce(.Add, sum1);
            output[row] = if (rounded) rows.round(v0) else v0;
            output[n + row] = if (rounded) rows.round(v1) else v1;
        }
        return;
    }

    for (lo..hi) |row| {
        const r_ptr = bytes + row * row_bytes;
        var first: usize = 0;
        while (first < count) : (first += 8) {
            const batch = @min(8, count - first);
            var sums: [8]@Vector(8, f32) = @splat(@splat(0));
            var b: usize = 0;
            while (b < blocks_per_row) : (b +%= 1) {
                const blk = r_ptr + b * 210;
                const d = readF16(blk + 208);
                inline for (0..2) |half| {
                    const ql_ptr = blk + half * 64;
                    const qh_ptr = blk + 128 + half * 32;
                    const sc_ptr = blk + 192 + half * 8;
                    const ql_0: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql_ptr)).*;
                    const ql_1: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql_ptr + 16)).*;
                    const ql_2: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql_ptr + 32)).*;
                    const ql_3: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(ql_ptr + 48)).*;
                    const qh_0: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qh_ptr)).*;
                    const qh_1: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qh_ptr + 16)).*;
                    inline for (0..4) |quarter| {
                        const ql_a = if (quarter % 2 == 0) ql_0 else ql_2;
                        const ql_b = if (quarter % 2 == 0) ql_1 else ql_3;
                        const c_a = q6kQuarter16(ql_a, qh_0, quarter);
                        const c_b = q6kQuarter16(ql_b, qh_1, quarter);
                        const s_a: @Vector(8, f32) = @splat(d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc_ptr[quarter * 2])))));
                        const s_b: @Vector(8, f32) = @splat(d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc_ptr[quarter * 2 + 1])))));
                        const w0 = s_a * i8x8ToF32x8(lowHalfI8(c_a));
                        const w1 = s_a * i8x8ToF32x8(highHalfI8(c_a));
                        const w2 = s_b * i8x8ToF32x8(lowHalfI8(c_b));
                        const w3 = s_b * i8x8ToF32x8(highHalfI8(c_b));
                        for (0..batch) |r| {
                            const xq = input + (first + r) * width + b * 256 + half * 128 + quarter * 32;
                            var acc = sums[r];
                            acc = @mulAdd(@Vector(8, f32), loadF32x8(xq), w0, acc);
                            acc = @mulAdd(@Vector(8, f32), loadF32x8(xq + 8), w1, acc);
                            acc = @mulAdd(@Vector(8, f32), loadF32x8(xq + 16), w2, acc);
                            acc = @mulAdd(@Vector(8, f32), loadF32x8(xq + 24), w3, acc);
                            sums[r] = acc;
                        }
                    }
                }
            }
            for (0..batch) |r| {
                const value = @reduce(.Add, sums[r]);
                output[(first + r) * n + row] = if (rounded) rows.round(value) else value;
            }
        }
    }
}

pub fn projectQ50Range(bytes: [*]const u8, input: [*]const f32, output: [*]f32, count: usize, width: usize, n: usize, rounded: bool, lo: usize, hi: usize) void {
    const blocks_per_row = width / 32;
    const row_bytes = blocks_per_row * 22;
    const mask4: @Vector(16, u8) = @splat(0x0F);
    const shift4: @Vector(16, u8) = @splat(4);
    const bias16: @Vector(8, i8) = @splat(16);

    if (count == 1) {
        var row = lo;
        while (row + 1 < hi) : (row +%= 2) {
            const r0_ptr = bytes + row * row_bytes;
            const r1_ptr = r0_ptr + row_bytes;
            var sum0: @Vector(8, f32) = @splat(0);
            var sum1: @Vector(8, f32) = @splat(0);
            var b: usize = 0;
            while (b < blocks_per_row) : (b +%= 1) {
                const b0 = r0_ptr + b * 22;
                const b1 = r1_ptr + b * 22;
                const s0: @Vector(8, f32) = @splat(readF16(b0));
                const s1: @Vector(8, f32) = @splat(readF16(b1));
                const qs0: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(b0 + 6)).*;
                const qs1: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(b1 + 6)).*;
                const lo0 = qs0 & mask4;
                const hi0 = qs0 >> shift4;
                const lo1 = qs1 & mask4;
                const hi1 = qs1 >> shift4;
                const xb = input + b * 32;

                const x0 = loadF32x8(xb);
                const c0_0: @Vector(8, i8) = @as(@Vector(8, i8), @bitCast(lowHalf(lo0) | expandBits8(b0[2]))) - bias16;
                const c1_0: @Vector(8, i8) = @as(@Vector(8, i8), @bitCast(lowHalf(lo1) | expandBits8(b1[2]))) - bias16;
                sum0 = @mulAdd(@Vector(8, f32), x0, s0 * i8x8ToF32x8(c0_0), sum0);
                sum1 = @mulAdd(@Vector(8, f32), x0, s1 * i8x8ToF32x8(c1_0), sum1);

                const x1 = loadF32x8(xb + 8);
                const c0_1: @Vector(8, i8) = @as(@Vector(8, i8), @bitCast(highHalf(lo0) | expandBits8(b0[3]))) - bias16;
                const c1_1: @Vector(8, i8) = @as(@Vector(8, i8), @bitCast(highHalf(lo1) | expandBits8(b1[3]))) - bias16;
                sum0 = @mulAdd(@Vector(8, f32), x1, s0 * i8x8ToF32x8(c0_1), sum0);
                sum1 = @mulAdd(@Vector(8, f32), x1, s1 * i8x8ToF32x8(c1_1), sum1);

                const x2 = loadF32x8(xb + 16);
                const c0_2: @Vector(8, i8) = @as(@Vector(8, i8), @bitCast(lowHalf(hi0) | expandBits8(b0[4]))) - bias16;
                const c1_2: @Vector(8, i8) = @as(@Vector(8, i8), @bitCast(lowHalf(hi1) | expandBits8(b1[4]))) - bias16;
                sum0 = @mulAdd(@Vector(8, f32), x2, s0 * i8x8ToF32x8(c0_2), sum0);
                sum1 = @mulAdd(@Vector(8, f32), x2, s1 * i8x8ToF32x8(c1_2), sum1);

                const x3 = loadF32x8(xb + 24);
                const c0_3: @Vector(8, i8) = @as(@Vector(8, i8), @bitCast(highHalf(hi0) | expandBits8(b0[5]))) - bias16;
                const c1_3: @Vector(8, i8) = @as(@Vector(8, i8), @bitCast(highHalf(hi1) | expandBits8(b1[5]))) - bias16;
                sum0 = @mulAdd(@Vector(8, f32), x3, s0 * i8x8ToF32x8(c0_3), sum0);
                sum1 = @mulAdd(@Vector(8, f32), x3, s1 * i8x8ToF32x8(c1_3), sum1);
            }
            const v0 = @reduce(.Add, sum0);
            const v1 = @reduce(.Add, sum1);
            output[row] = if (rounded) rows.round(v0) else v0;
            output[row + 1] = if (rounded) rows.round(v1) else v1;
        }
        if (row < hi) {
            const r0_ptr = bytes + row * row_bytes;
            var sum0: @Vector(8, f32) = @splat(0);
            var b: usize = 0;
            while (b < blocks_per_row) : (b +%= 1) {
                const b0 = r0_ptr + b * 22;
                const s0: @Vector(8, f32) = @splat(readF16(b0));
                const qs0: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(b0 + 6)).*;
                const lo0 = qs0 & mask4;
                const hi0 = qs0 >> shift4;
                const xb = input + b * 32;
                sum0 = @mulAdd(@Vector(8, f32), loadF32x8(xb), s0 * i8x8ToF32x8(@as(@Vector(8, i8), @bitCast(lowHalf(lo0) | expandBits8(b0[2]))) - bias16), sum0);
                sum0 = @mulAdd(@Vector(8, f32), loadF32x8(xb + 8), s0 * i8x8ToF32x8(@as(@Vector(8, i8), @bitCast(highHalf(lo0) | expandBits8(b0[3]))) - bias16), sum0);
                sum0 = @mulAdd(@Vector(8, f32), loadF32x8(xb + 16), s0 * i8x8ToF32x8(@as(@Vector(8, i8), @bitCast(lowHalf(hi0) | expandBits8(b0[4]))) - bias16), sum0);
                sum0 = @mulAdd(@Vector(8, f32), loadF32x8(xb + 24), s0 * i8x8ToF32x8(@as(@Vector(8, i8), @bitCast(highHalf(hi0) | expandBits8(b0[5]))) - bias16), sum0);
            }
            const v0 = @reduce(.Add, sum0);
            output[row] = if (rounded) rows.round(v0) else v0;
        }
        return;
    }

    for (lo..hi) |row| {
        const r_ptr = bytes + row * row_bytes;
        var first: usize = 0;
        while (first < count) : (first += 8) {
            const batch = @min(8, count - first);
            var sums: [8]@Vector(8, f32) = @splat(@splat(0));
            var b: usize = 0;
            while (b < blocks_per_row) : (b +%= 1) {
                const blk = r_ptr + b * 22;
                const s: @Vector(8, f32) = @splat(readF16(blk));
                const qs: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(blk + 6)).*;
                const lo_n = qs & mask4;
                const hi_n = qs >> shift4;
                const w0 = s * i8x8ToF32x8(@as(@Vector(8, i8), @bitCast(lowHalf(lo_n) | expandBits8(blk[2]))) - bias16);
                const w1 = s * i8x8ToF32x8(@as(@Vector(8, i8), @bitCast(highHalf(lo_n) | expandBits8(blk[3]))) - bias16);
                const w2 = s * i8x8ToF32x8(@as(@Vector(8, i8), @bitCast(lowHalf(hi_n) | expandBits8(blk[4]))) - bias16);
                const w3 = s * i8x8ToF32x8(@as(@Vector(8, i8), @bitCast(highHalf(hi_n) | expandBits8(blk[5]))) - bias16);
                for (0..batch) |r| {
                    const xb = input + (first + r) * width + b * 32;
                    var acc = sums[r];
                    acc = @mulAdd(@Vector(8, f32), loadF32x8(xb), w0, acc);
                    acc = @mulAdd(@Vector(8, f32), loadF32x8(xb + 8), w1, acc);
                    acc = @mulAdd(@Vector(8, f32), loadF32x8(xb + 16), w2, acc);
                    acc = @mulAdd(@Vector(8, f32), loadF32x8(xb + 24), w3, acc);
                    sums[r] = acc;
                }
            }
            for (0..batch) |r| {
                const value = @reduce(.Add, sums[r]);
                output[(first + r) * n + row] = if (rounded) rows.round(value) else value;
            }
        }
    }
}
