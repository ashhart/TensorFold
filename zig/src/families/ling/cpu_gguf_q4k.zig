//! Exact Q4_K rows: each dot keeps the eight-lane FMA order of the scalar reference, so results match it bit for bit.
const rows = @import("cpu_rows.zig");
const lanes = @import("cpu_gguf_lanes.zig");
const readF16 = lanes.readF16;
const loadF32x8 = lanes.loadF32x8;
const lowHalf = lanes.lowHalf;
const highHalf = lanes.highHalf;
const u8x8ToF32x8 = lanes.u8x8ToF32x8;

inline fn decodeQ4KScales(b: [*]const u8, scales: *[8]f32, biases: *[8]f32) void {
    const d = readF16(b);
    const m = readF16(b + 2);
    const q = b + 4;
    inline for (0..4) |grp| {
        const s: u8 = q[grp] & 63;
        const mn: u8 = q[grp + 4] & 63;
        scales[grp] = d * @as(f32, @floatFromInt(s));
        biases[grp] = -m * @as(f32, @floatFromInt(mn));
    }
    inline for (4..8) |grp| {
        const s: u8 = (q[grp + 4] & 15) | ((q[grp - 4] >> 6) << 4);
        const mn: u8 = (q[grp + 4] >> 4) | ((q[grp] >> 6) << 4);
        scales[grp] = d * @as(f32, @floatFromInt(s));
        biases[grp] = -m * @as(f32, @floatFromInt(mn));
    }
}

pub fn projectQ4KRange(bytes: [*]const u8, input: [*]const f32, output: [*]f32, count: usize, width: usize, n: usize, rounded: bool, lo: usize, hi: usize) void {
    const blocks_per_row = width / 256;
    const row_bytes = blocks_per_row * 144;
    const mask4: @Vector(16, u8) = @splat(0x0F);
    const shift4: @Vector(16, u8) = @splat(4);

    if (count == 1) {
        var row = lo;
        // Process 2 output rows in parallel to break FMA latency chains and reuse L1 input loads.
        while (row + 1 < hi) : (row +%= 2) {
            const r0_ptr = bytes + row * row_bytes;
            const r1_ptr = r0_ptr + row_bytes;
            var sum0: @Vector(8, f32) = @splat(0);
            var sum1: @Vector(8, f32) = @splat(0);
            var b: usize = 0;
            while (b < blocks_per_row) : (b +%= 1) {
                const b0 = r0_ptr + b * 144;
                const b1 = r1_ptr + b * 144;
                var scales0: [8]f32 = undefined;
                var biases0: [8]f32 = undefined;
                var scales1: [8]f32 = undefined;
                var biases1: [8]f32 = undefined;
                decodeQ4KScales(b0, &scales0, &biases0);
                decodeQ4KScales(b1, &scales1, &biases1);
                const x_blk = input + b * 256;

                inline for (0..4) |chunk| {
                    const qs0 = b0 + 16 + chunk * 32;
                    const qs1 = b1 + 16 + chunk * 32;
                    const raw0_a: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qs0)).*;
                    const raw0_b: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qs0 + 16)).*;
                    const raw1_a: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qs1)).*;
                    const raw1_b: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qs1 + 16)).*;
                    const xc = x_blk + chunk * 64;

                    // Low nibbles (group 2 * chunk)
                    {
                        const lo0_a = raw0_a & mask4;
                        const lo0_b = raw0_b & mask4;
                        const lo1_a = raw1_a & mask4;
                        const lo1_b = raw1_b & mask4;
                        const s0: @Vector(8, f32) = @splat(scales0[2 * chunk]);
                        const k0: @Vector(8, f32) = @splat(biases0[2 * chunk]);
                        const s1: @Vector(8, f32) = @splat(scales1[2 * chunk]);
                        const k1: @Vector(8, f32) = @splat(biases1[2 * chunk]);

                        const x0 = loadF32x8(xc);
                        sum0 = @mulAdd(@Vector(8, f32), x0, s0 * u8x8ToF32x8(lowHalf(lo0_a)) + k0, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), x0, s1 * u8x8ToF32x8(lowHalf(lo1_a)) + k1, sum1);

                        const x1 = loadF32x8(xc + 8);
                        sum0 = @mulAdd(@Vector(8, f32), x1, s0 * u8x8ToF32x8(highHalf(lo0_a)) + k0, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), x1, s1 * u8x8ToF32x8(highHalf(lo1_a)) + k1, sum1);

                        const x2 = loadF32x8(xc + 16);
                        sum0 = @mulAdd(@Vector(8, f32), x2, s0 * u8x8ToF32x8(lowHalf(lo0_b)) + k0, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), x2, s1 * u8x8ToF32x8(lowHalf(lo1_b)) + k1, sum1);

                        const x3 = loadF32x8(xc + 24);
                        sum0 = @mulAdd(@Vector(8, f32), x3, s0 * u8x8ToF32x8(highHalf(lo0_b)) + k0, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), x3, s1 * u8x8ToF32x8(highHalf(lo1_b)) + k1, sum1);
                    }
                    // High nibbles (group 2 * chunk + 1)
                    {
                        const hi0_a = raw0_a >> shift4;
                        const hi0_b = raw0_b >> shift4;
                        const hi1_a = raw1_a >> shift4;
                        const hi1_b = raw1_b >> shift4;
                        const s0: @Vector(8, f32) = @splat(scales0[2 * chunk + 1]);
                        const k0: @Vector(8, f32) = @splat(biases0[2 * chunk + 1]);
                        const s1: @Vector(8, f32) = @splat(scales1[2 * chunk + 1]);
                        const k1: @Vector(8, f32) = @splat(biases1[2 * chunk + 1]);

                        const x4 = loadF32x8(xc + 32);
                        sum0 = @mulAdd(@Vector(8, f32), x4, s0 * u8x8ToF32x8(lowHalf(hi0_a)) + k0, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), x4, s1 * u8x8ToF32x8(lowHalf(hi1_a)) + k1, sum1);

                        const x5 = loadF32x8(xc + 40);
                        sum0 = @mulAdd(@Vector(8, f32), x5, s0 * u8x8ToF32x8(highHalf(hi0_a)) + k0, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), x5, s1 * u8x8ToF32x8(highHalf(hi1_a)) + k1, sum1);

                        const x6 = loadF32x8(xc + 48);
                        sum0 = @mulAdd(@Vector(8, f32), x6, s0 * u8x8ToF32x8(lowHalf(hi0_b)) + k0, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), x6, s1 * u8x8ToF32x8(lowHalf(hi1_b)) + k1, sum1);

                        const x7 = loadF32x8(xc + 56);
                        sum0 = @mulAdd(@Vector(8, f32), x7, s0 * u8x8ToF32x8(highHalf(hi0_b)) + k0, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), x7, s1 * u8x8ToF32x8(highHalf(hi1_b)) + k1, sum1);
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
                const b0 = r0_ptr + b * 144;
                var scales0: [8]f32 = undefined;
                var biases0: [8]f32 = undefined;
                decodeQ4KScales(b0, &scales0, &biases0);
                const x_blk = input + b * 256;
                inline for (0..4) |chunk| {
                    const qs0 = b0 + 16 + chunk * 32;
                    const raw0_a: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qs0)).*;
                    const raw0_b: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qs0 + 16)).*;
                    const xc = x_blk + chunk * 64;
                    {
                        const lo0_a = raw0_a & mask4;
                        const lo0_b = raw0_b & mask4;
                        const s0: @Vector(8, f32) = @splat(scales0[2 * chunk]);
                        const k0: @Vector(8, f32) = @splat(biases0[2 * chunk]);
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(xc), s0 * u8x8ToF32x8(lowHalf(lo0_a)) + k0, sum0);
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(xc + 8), s0 * u8x8ToF32x8(highHalf(lo0_a)) + k0, sum0);
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(xc + 16), s0 * u8x8ToF32x8(lowHalf(lo0_b)) + k0, sum0);
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(xc + 24), s0 * u8x8ToF32x8(highHalf(lo0_b)) + k0, sum0);
                    }
                    {
                        const hi0_a = raw0_a >> shift4;
                        const hi0_b = raw0_b >> shift4;
                        const s0: @Vector(8, f32) = @splat(scales0[2 * chunk + 1]);
                        const k0: @Vector(8, f32) = @splat(biases0[2 * chunk + 1]);
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(xc + 32), s0 * u8x8ToF32x8(lowHalf(hi0_a)) + k0, sum0);
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(xc + 40), s0 * u8x8ToF32x8(highHalf(hi0_a)) + k0, sum0);
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(xc + 48), s0 * u8x8ToF32x8(lowHalf(hi0_b)) + k0, sum0);
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(xc + 56), s0 * u8x8ToF32x8(highHalf(hi0_b)) + k0, sum0);
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
                const blk = r_ptr + b * 144;
                var scales: [8]f32 = undefined;
                var biases: [8]f32 = undefined;
                decodeQ4KScales(blk, &scales, &biases);
                const x0_blk = input + b * 256;
                const x1_blk = input1 + b * 256;
                inline for (0..4) |chunk| {
                    const qs = blk + 16 + chunk * 32;
                    const raw_a: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qs)).*;
                    const raw_b: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qs + 16)).*;
                    const x0c = x0_blk + chunk * 64;
                    const x1c = x1_blk + chunk * 64;
                    {
                        const lo_a = raw_a & mask4;
                        const lo_b = raw_b & mask4;
                        const s_lo: @Vector(8, f32) = @splat(scales[2 * chunk]);
                        const b_lo: @Vector(8, f32) = @splat(biases[2 * chunk]);
                        const w0 = s_lo * u8x8ToF32x8(lowHalf(lo_a)) + b_lo;
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(x0c), w0, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), loadF32x8(x1c), w0, sum1);
                        const w1 = s_lo * u8x8ToF32x8(highHalf(lo_a)) + b_lo;
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(x0c + 8), w1, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), loadF32x8(x1c + 8), w1, sum1);
                        const w2 = s_lo * u8x8ToF32x8(lowHalf(lo_b)) + b_lo;
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(x0c + 16), w2, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), loadF32x8(x1c + 16), w2, sum1);
                        const w3 = s_lo * u8x8ToF32x8(highHalf(lo_b)) + b_lo;
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(x0c + 24), w3, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), loadF32x8(x1c + 24), w3, sum1);
                    }
                    {
                        const hi_a = raw_a >> shift4;
                        const hi_b = raw_b >> shift4;
                        const s_hi: @Vector(8, f32) = @splat(scales[2 * chunk + 1]);
                        const b_hi: @Vector(8, f32) = @splat(biases[2 * chunk + 1]);
                        const w4 = s_hi * u8x8ToF32x8(lowHalf(hi_a)) + b_hi;
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(x0c + 32), w4, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), loadF32x8(x1c + 32), w4, sum1);
                        const w5 = s_hi * u8x8ToF32x8(highHalf(hi_a)) + b_hi;
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(x0c + 40), w5, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), loadF32x8(x1c + 40), w5, sum1);
                        const w6 = s_hi * u8x8ToF32x8(lowHalf(hi_b)) + b_hi;
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(x0c + 48), w6, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), loadF32x8(x1c + 48), w6, sum1);
                        const w7 = s_hi * u8x8ToF32x8(highHalf(hi_b)) + b_hi;
                        sum0 = @mulAdd(@Vector(8, f32), loadF32x8(x0c + 56), w7, sum0);
                        sum1 = @mulAdd(@Vector(8, f32), loadF32x8(x1c + 56), w7, sum1);
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
                const blk = r_ptr + b * 144;
                var scales: [8]f32 = undefined;
                var biases: [8]f32 = undefined;
                decodeQ4KScales(blk, &scales, &biases);
                inline for (0..4) |chunk| {
                    const qs = blk + 16 + chunk * 32;
                    const raw_a: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qs)).*;
                    const raw_b: @Vector(16, u8) = @as(*align(1) const [16]u8, @ptrCast(qs + 16)).*;
                    const lo_a = raw_a & mask4;
                    const lo_b = raw_b & mask4;
                    const s_lo: @Vector(8, f32) = @splat(scales[2 * chunk]);
                    const b_lo: @Vector(8, f32) = @splat(biases[2 * chunk]);
                    const w0 = s_lo * u8x8ToF32x8(lowHalf(lo_a)) + b_lo;
                    const w1 = s_lo * u8x8ToF32x8(highHalf(lo_a)) + b_lo;
                    const w2 = s_lo * u8x8ToF32x8(lowHalf(lo_b)) + b_lo;
                    const w3 = s_lo * u8x8ToF32x8(highHalf(lo_b)) + b_lo;

                    const hi_a = raw_a >> shift4;
                    const hi_b = raw_b >> shift4;
                    const s_hi: @Vector(8, f32) = @splat(scales[2 * chunk + 1]);
                    const b_hi: @Vector(8, f32) = @splat(biases[2 * chunk + 1]);
                    const w4 = s_hi * u8x8ToF32x8(lowHalf(hi_a)) + b_hi;
                    const w5 = s_hi * u8x8ToF32x8(highHalf(hi_a)) + b_hi;
                    const w6 = s_hi * u8x8ToF32x8(lowHalf(hi_b)) + b_hi;
                    const w7 = s_hi * u8x8ToF32x8(highHalf(hi_b)) + b_hi;

                    for (0..batch) |r| {
                        const xc = input + (first + r) * width + b * 256 + chunk * 64;
                        var acc = sums[r];
                        acc = @mulAdd(@Vector(8, f32), loadF32x8(xc), w0, acc);
                        acc = @mulAdd(@Vector(8, f32), loadF32x8(xc + 8), w1, acc);
                        acc = @mulAdd(@Vector(8, f32), loadF32x8(xc + 16), w2, acc);
                        acc = @mulAdd(@Vector(8, f32), loadF32x8(xc + 24), w3, acc);
                        acc = @mulAdd(@Vector(8, f32), loadF32x8(xc + 32), w4, acc);
                        acc = @mulAdd(@Vector(8, f32), loadF32x8(xc + 40), w5, acc);
                        acc = @mulAdd(@Vector(8, f32), loadF32x8(xc + 48), w6, acc);
                        acc = @mulAdd(@Vector(8, f32), loadF32x8(xc + 56), w7, acc);
                        sums[r] = acc;
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
