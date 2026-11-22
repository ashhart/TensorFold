//! Load-time sm_121a repackers for Inkling 2 Mini (Stage 3 spec).
//! Pure host-side math; the CUDA side lives in kernels/cuda/inkling2/sm121a_moe.cu.
//!   - NVFP4 -> OMMA packExpert cells ([E][N/32][K/32][M][36 uint4]) + amax/den fold into scale2.
//!   - MXFP8 -> QMMA fragment order (w8) + tiled exponents (bs).
//!   - BF16 dense linears -> lossless BF12 tiles (port of flashnext/cuda_bf12.zig).
const std = @import("std");

pub const tile_bytes: usize = 768;

// ---- NVFP4 ------------------------------------------------------------------

/// One expert matrix [n_full, k] row-major nibbles/scales -> out[cb][g] cells of 144 u32
/// (m-th matrix of m_count at cell offset m*144), strided by (row0, row_step) for interleaved w13.
pub fn packNvfp4Expert(
    codes: []const u8, // [n_full * k/2] bytes, low nibble first
    scales: []const u8, // [n_full * k/16] e4m3 bytes
    n: usize, // packed output rows (n_full / row_step)
    k: usize,
    row0: usize,
    row_step: usize,
    out: []u32, // [(n/32) * (k/32) * m_count * 144]
    m: usize,
    m_count: usize,
) void {
    const nb = n / 32;
    const kg = k / 32;
    std.debug.assert(out.len == nb * kg * m_count * 144);
    const word_pitch = k / 2;
    const scale_pitch = k / 16;
    for (0..nb) |cb| {
        for (0..kg) |g| {
            const cell = out[((cb * kg + g) * m_count + m) * 144 ..][0..144];
            for (0..8) |gq| {
                for (0..4) |j| {
                    const r = row0 + row_step * (cb * 32 + j * 8 + gq);
                    const row = codes[r * word_pitch + g * 16 ..][0..16];
                    for (0..4) |t| {
                        const b0: u32 = row[2 * t];
                        const b1: u32 = row[2 * t + 1];
                        const b2: u32 = row[8 + 2 * t];
                        const b3: u32 = row[8 + 2 * t + 1];
                        const lo = (b0 & 0xF) | (b1 & 0xF) << 4 | (b2 & 0xF) << 8 | (b3 & 0xF) << 12;
                        const hi = (b0 >> 4) | (b1 >> 4) << 4 | (b2 >> 4) << 8 | (b3 >> 4) << 12;
                        cell[(gq * 4 + t) * 4 + j] = lo | hi << 16;
                    }
                }
            }
            const bytes = std.mem.sliceAsBytes(cell[128..144]);
            for (0..4) |t| {
                for (0..4) |j| {
                    for (0..2) |c| {
                        const r = row0 + row_step * (cb * 32 + j * 8 + t * 2 + c);
                        const sc = scales[r * scale_pitch + g * 2 ..][0..2];
                        bytes[t * 16 + j * 2 + c] = sc[0];
                        bytes[t * 16 + 8 + j * 2 + c] = sc[1];
                    }
                }
            }
        }
    }
}

fn bfRound(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    const rounded: u32 = (bits +% 0x7FFF +% ((bits >> 16) & 1)) & 0xFFFF0000;
    return @bitCast(rounded);
}

/// Fold the NVFP4 outer scale into the OMMA per-matrix scale2, matching modelopt.cu:45 bf(amax)/bf(den).
pub fn foldNvfp4Outer(amax: f32, den: f32) f32 {
    return bfRound(amax) / bfRound(den);
}

// ---- MXFP8 ------------------------------------------------------------------

/// Input offset within a 32-input step of fragment byte `byte` of `lane`.
pub fn kin(lane: usize, byte: usize) usize {
    return 16 * (byte / 4) + 2 * (lane % 4) + (byte % 2) + 8 * ((byte % 4) / 2);
}

/// e4m3 bytes [n_full, k] -> [npad/64][k/64][8][32][2][8] fragment order, strided by (row0, row_step).
pub fn fragmentOrder(codes: []const u8, n: usize, k: usize, npad: usize, row0: usize, row_step: usize, out: []u8) void {
    std.debug.assert(k % 64 == 0 and npad % 64 == 0 and npad >= n);
    std.debug.assert(out.len == npad * k);
    var i: usize = 0;
    for (0..npad / 64) |a| {
        for (0..k / 64) |b| {
            for (0..8) |c| {
                for (0..32) |lane| {
                    for (0..2) |h| {
                        for (0..8) |byte| {
                            const r = a * 64 + c * 8 + lane / 4;
                            const row = row0 + row_step * r;
                            const col = b * 64 + h * 32 + kin(lane, byte);
                            out[i] = if (r < n) codes[row * k + col] else 0;
                            i += 1;
                        }
                    }
                }
            }
        }
    }
}

/// e8m0 bytes [n_full, k/32] -> [npad/64][k/64][64][2], strided by (row0, row_step).
pub fn tileExponents(exps: []const u8, n: usize, npad: usize, k: usize, row0: usize, row_step: usize, out: []u8) void {
    const kg = k / 64;
    for (0..npad / 64) |a| {
        for (0..kg) |g| {
            for (0..64) |r| {
                const rr = a * 64 + r;
                for (0..2) |h| {
                    out[((a * kg + g) * 64 + r) * 2 + h] = if (rr < n) exps[(row0 + row_step * rr) * (k / 32) + g * 2 + h] else 0;
                }
            }
        }
    }
}

// ---- BF12 -------------------------------------------------------------------

fn bestBase(row: []const u16) u8 {
    var hist: [256]u32 = @splat(0);
    for (row) |v| {
        const e = (v >> 7) & 0xFF;
        if (e >= 1 and e <= 254) hist[e] += 1;
    }
    var cur: u32 = 0;
    for (1..16) |e| cur += hist[e];
    var best = cur;
    var best_b: u8 = if (cur > 0) 1 else 112;
    for (2..241) |b| {
        cur += hist[b + 14];
        cur -= hist[b - 1];
        if (cur > best) {
            best = cur;
            best_b = @intCast(b);
        }
    }
    return best_b;
}

fn encode(v: u16, base: u8, sm: *u8, ec: *u8) void {
    const sign: u8 = @intCast(v >> 15);
    const e: u32 = (v >> 7) & 0xFF;
    const mant: u8 = @intCast(v & 0x7F);
    if (e >= base and e <= @as(u32, base) + 14) {
        ec.* = @intCast(e - base);
        sm.* = (sign << 7) | mant;
    } else if (e == 0 and mant == 0) {
        ec.* = 15;
        sm.* = sign << 7;
    } else {
        ec.* = 15;
        sm.* = 0x01;
    }
}

pub fn decode(sm: u8, ec: u8, base: u8) u16 {
    if (ec < 15) return (@as(u16, sm & 0x80) << 8) | (@as(u16, base + ec) << 7) | (sm & 0x7F);
    if (sm & 0x7F == 0) return @as(u16, sm & 0x80) << 8;
    return 0;
}

/// Host-only BF12 pack of one bf16 [n,k] (k % 64 == 0): tiles + per-row bases +
/// escape entries. Mirrors cuda_bf12.zig:pack without the device round-trip.
pub const Bf12Packed = struct {
    data: []u8, // [((n+7)/8) * (k/64) * 768]
    row_base: []u8, // [((n+7)/8) * 8]
    row_ptr: []u32, // [n + 1] escape offsets per row
    entries: []u32, // caller buffer; escapes as (col<<16)|v
    n_entries: usize,
};

/// Where tile (8-row group `n8`, 64-column group `kg`) lives: the kernels' [64-row block][kg][tile in block] order
/// (bf12.cu unpack and mm), a short last block holding only its own tiles. Not [n8][kg].
pub fn tileOffset(n8: usize, kg: usize, kg_n: usize, total_n8: usize) usize {
    const b = n8 / 8;
    const block_n8 = @min(8, total_n8 - b * 8);
    return b * kg_n * 8 * tile_bytes + kg * block_n8 * tile_bytes + (n8 % 8) * tile_bytes;
}

pub fn packBf12(rows: []const u16, n: usize, k: usize, out: Bf12Packed) !usize {
    std.debug.assert(k % 64 == 0 and rows.len == n * k);
    std.debug.assert(out.row_ptr.len == n + 1);
    const total_n8 = (n + 7) / 8;
    const kg_n = k / 64;
    std.debug.assert(out.data.len == total_n8 * kg_n * tile_bytes);
    @memset(out.data, 0);
    out.row_ptr[0] = 0;
    var n_entries: usize = 0;
    var n0: usize = 0;
    while (n0 < n) : (n0 += 64) {
        const n8_0 = n0 / 8;
        const rows_here = @min(64, n - n0);
        const n8_c = (rows_here + 7) / 8;
        for (0..n8_c) |t| {
            for (0..8) |li| {
                const r = n0 + t * 8 + li;
                if (r < n) out.row_base[n8_0 * 8 + t * 8 + li] = bestBase(rows[r * k ..][0..k]);
            }
        }
        for (0..kg_n) |kg| {
            const blk_n8 = @min(8, total_n8 - n8_0);
            for (0..blk_n8) |t| {
                const tile = out.data[tileOffset(n8_0 + t, kg, kg_n, total_n8)..][0..tile_bytes];
                for (0..32) |lane| {
                    const li = t * 8 + (lane >> 2);
                    const tid4 = lane & 3;
                    const have = li < rows_here;
                    const base = out.row_base[(n8_0 + t) * 8 + (lane >> 2)];
                    for (0..4) |kt| {
                        const kb = kg * 64 + kt * 16 + tid4 * 2;
                        var sm = [4]u8{ 0, 0, 0, 0 };
                        if (have) for ([4]usize{ kb, kb + 1, kb + 8, kb + 9 }, 0..) |c, j| {
                            const v = rows[(n0 + li) * k + c];
                            var s: u8 = 0;
                            var e: u8 = 15;
                            encode(v, base, &s, &e);
                            sm[j] = s;
                        };
                        const word = @as(u32, sm[0]) | (@as(u32, sm[2]) << 8) | (@as(u32, sm[1]) << 16) | (@as(u32, sm[3]) << 24);
                        std.mem.writeInt(u32, tile[(lane * 4 + kt) * 4 ..][0..4], word, .little);
                    }
                    // ec words: recompute per element for the 512..768 region
                    var ec_words = [2]u32{ 0, 0 };
                    for (0..4) |kt| {
                        const kb = kg * 64 + kt * 16 + tid4 * 2;
                        var ec = [4]u8{ 15, 15, 15, 15 };
                        if (have) for ([4]usize{ kb, kb + 1, kb + 8, kb + 9 }, 0..) |c, j| {
                            const v = rows[(n0 + li) * k + c];
                            var s: u8 = 0;
                            var e: u8 = 15;
                            encode(v, base, &s, &e);
                            ec[j] = e;
                        };
                        const sh: u5 = @intCast((kt & 1) * 8);
                        ec_words[kt >> 1] |= (@as(u32, ec[0]) << sh) | (@as(u32, ec[2]) << (sh + 4)) | (@as(u32, ec[1]) << (sh + 16)) | (@as(u32, ec[3]) << (sh + 20));
                    }
                    std.mem.writeInt(u32, tile[512 + lane * 8 ..][0..4], ec_words[0], .little);
                    std.mem.writeInt(u32, tile[512 + lane * 8 + 4 ..][0..4], ec_words[1], .little);
                }
            }
        }
        // escapes for this 64-row block
        for (0..rows_here) |li| {
            const base = out.row_base[n8_0 * 8 + li];
            for (rows[(n0 + li) * k ..][0..k], 0..) |v, c| {
                const e = (v >> 7) & 0xFF;
                if (!(e >= base and e <= @as(u32, base) + 14) and (v & 0x7FFF) != 0) {
                    if (n_entries >= out.entries.len) return error.Bf12EntriesOverflow;
                    out.entries[n_entries] = (@as(u32, @intCast(c)) << 16) | v;
                    n_entries += 1;
                }
            }
            out.row_ptr[n0 + li + 1] = @intCast(n_entries);
        }
    }
    return n_entries;
}

// ---- tests ------------------------------------------------------------------

test "nvfp4 packExpert cell layout: codes and scale bytes land per spec" {
    const n = 32;
    const k = 64;
    var codes: [32 * 32]u8 = undefined; // k/2 bytes a row
    var scales: [32 * 4]u8 = undefined; // k/16 bytes a row
    for (&codes, 0..) |*b, i| b.* = @intCast(i & 0xFF);
    for (&scales, 0..) |*b, i| b.* = @intCast((i * 7) & 0xFF);
    var out: [1 * 2 * 1 * 144]u32 = undefined;
    packNvfp4Expert(&codes, &scales, n, k, 0, 1, &out, 0, 1);
    // first cell word: gq=0,j=0,t=0 -> rows r=0, bytes row[0],row[1],row[8],row[9]
    const c0: u32 = codes[0];
    const c1: u32 = codes[1];
    const c8: u32 = codes[8];
    const c9: u32 = codes[9];
    const lo = (c0 & 0xF) | (c1 & 0xF) << 4 | (c8 & 0xF) << 8 | (c9 & 0xF) << 12;
    const hi = (c0 >> 4) | (c1 >> 4) << 4 | (c8 >> 4) << 8 | (c9 >> 4) << 12;
    try std.testing.expectEqual(lo | hi << 16, out[0]);
    // scale bytes: cell[128..144] as bytes; byte[t*16+j*2+c], byte[t*16+8+j*2+c]
    const bytes = std.mem.sliceAsBytes(out[128..144]);
    try std.testing.expectEqual(scales[0], bytes[0]); // t=0,j=0,c=0 -> r=0, sc[0]=scales[0]
    try std.testing.expectEqual(scales[4], bytes[1]); // t=0,j=0,c=1 -> r=1, sc[0]=scales[1*4]
    try std.testing.expectEqual(scales[1], bytes[8]); // second half: sc[1] of r=0
    // amax/den fold
    try std.testing.expectEqual(@as(f32, 2.5), foldNvfp4Outer(5.0, 2.0));
}

test "mxfp8 fragmentOrder/kin round-trips; tileExponents matches doc example" {
    // doc example from cuda_fp8 test
    var e: [128 * 4]u8 = undefined;
    for (&e, 0..) |*v, i| v.* = @intCast(i % 251);
    var tiled: [128 * 4]u8 = undefined;
    tileExponents(&e, 128, 128, 128, 0, 1, &tiled);
    try std.testing.expectEqual(e[69 * 4 + 3], tiled[((1 * 2 + 1) * 64 + 5) * 2 + 1]);
    try std.testing.expectEqual(e[2], tiled[((0 * 2 + 1) * 64 + 0) * 2 + 0]);
    // round-trip: fragment then gather back
    const n = 64;
    const k = 128;
    var codes: [64 * 128]u8 = undefined;
    for (&codes, 0..) |*v, i| v.* = @intCast((i * 13 + 7) & 0xFF);
    var frag: [64 * 128]u8 = undefined;
    fragmentOrder(&codes, n, k, 64, 0, 1, &frag);
    var back: [64 * 128]u8 = @splat(0);
    var i: usize = 0;
    for (0..1) |a| {
        for (0..k / 64) |b| {
            for (0..8) |c| {
                for (0..32) |lane| {
                    for (0..2) |h| {
                        for (0..8) |byte| {
                            const row = a * 64 + c * 8 + lane / 4;
                            const col = b * 64 + h * 32 + kin(lane, byte);
                            back[row * k + col] = frag[i];
                            i += 1;
                        }
                    }
                }
            }
        }
    }
    try std.testing.expectEqualSlices(u8, &codes, &back);
}

test "bf12 element code round-trips all 65536 patterns" {
    var v: u32 = 0;
    while (v < 65536) : (v += 1) {
        const w: u16 = @intCast(v);
        const e = (w >> 7) & 0xFF;
        const base: u8 = if (e >= 1 and e <= 240) @intCast(e) else 112;
        var sm: u8 = 0;
        var ec: u8 = 0;
        encode(w, base, &sm, &ec);
        const in_window = e >= base and e <= @as(u32, base) + 14;
        const zero = (w & 0x7FFF) == 0;
        if (in_window or zero) {
            try std.testing.expectEqual(w, decode(sm, ec, base));
        } else {
            try std.testing.expect(ec == 15 and sm == 0x01);
        }
    }
}

test "bf12 pack:lossless rows decode back; escapes match entries" {
    const n = 16;
    const k = 128;
    var rows: [16 * 128]u16 = undefined;
    var prng = std.Random.DefaultPrng.init(42);
    const rnd = prng.random();
    for (&rows) |*v| {
        // bf16-ish values: random exponent in a narrow band (mostly in-window) + noise
        const e: u16 = if (rnd.boolean()) 100 + rnd.intRangeLessThan(u16, 0, 10) else 200;
        v.* = (@as(u16, rnd.int(u1)) << 15) | (e << 7) | rnd.int(u7);
    }
    var data: [2 * 2 * 768]u8 = undefined; // total_n8=2, kg=2
    var bases: [16]u8 = undefined;
    var row_ptr: [17]u32 = undefined;
    var entries: [16 * 128]u32 = undefined;
    const out = Bf12Packed{ .data = &data, .row_base = &bases, .row_ptr = &row_ptr, .entries = &entries, .n_entries = 0 };
    const n_packed = try packBf12(&rows, n, k, out);
    try std.testing.expectEqual(@as(u32, 0), row_ptr[0]);
    try std.testing.expectEqual(@as(u32, @intCast(n_packed)), row_ptr[n]);
    // every element either decodes exactly from its tile or appears in entries
    const kg_n = k / 64;
    var n_esc: usize = 0;
    for (0..n) |r| {
        const base = bases[(r / 8) * 8 + r % 8];
        for (0..k) |c| {
            const v = rows[r * k + c];
            const e = (v >> 7) & 0xFF;
            if ((e >= base and e <= @as(u32, base) + 14) or (v & 0x7FFF) == 0) {
                // locate packed sm/ec: block n8=r/8, group kg=c/64; within the
                // 64-col group: kt=(c%64)/16, d=(c%64)%16, j=(d/8)*2+(d%2).
                const n8 = r / 8;
                const li = r % 8;
                const kg = c / 64;
                const tile = data[tileOffset(n8, kg, kg_n, 2)..][0..tile_bytes];
                const c64 = c % 64;
                const kt = c64 / 16;
                const d = c64 % 16;
                const jj: usize = (d / 8) * 2 + (d % 2);
                const lane = li * 4 + (d % 8) / 2;
                const word = std.mem.readInt(u32, tile[(lane * 4 + kt) * 4 ..][0..4], .little);
                const sm_byte: u8 = switch (jj) {
                    0 => @intCast(word & 0xFF),
                    1 => @intCast((word >> 16) & 0xFF),
                    2 => @intCast((word >> 8) & 0xFF),
                    3 => @intCast((word >> 24) & 0xFF),
                    else => unreachable,
                };
                const ecw = std.mem.readInt(u32, tile[512 + lane * 8 + (kt >> 1) * 4 ..][0..4], .little);
                const ec_off = [4]u8{ 0, 16, 4, 20 };
                const sh: u5 = @intCast((kt & 1) * 8 + ec_off[jj]);
                const ecv: u8 = @intCast((ecw >> sh) & 0xF);
                try std.testing.expectEqual(v, decode(sm_byte, ecv, base));
            } else {
                n_esc += 1;
            }
        }
    }
    try std.testing.expectEqual(n_esc, n_packed);
}

test "bf12 pack: tiles land where the kernels read them across several 64-row blocks and a short last block" {
    const n = 72; // total_n8 = 9: one block of 8 tiles, one of 1
    const k = 192; // kg = 3
    const gpa = std.testing.allocator;
    const rows = try gpa.alloc(u16, n * k);
    defer gpa.free(rows);
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for (rows) |*v| v.* = (@as(u16, rnd.int(u1)) << 15) | ((100 + rnd.intRangeLessThan(u16, 0, 12)) << 7) | rnd.int(u7);
    const total_n8 = (n + 7) / 8;
    const kg_n = k / 64;
    const data = try gpa.alloc(u8, total_n8 * kg_n * tile_bytes);
    defer gpa.free(data);
    const bases = try gpa.alloc(u8, total_n8 * 8);
    defer gpa.free(bases);
    const row_ptr = try gpa.alloc(u32, n + 1);
    defer gpa.free(row_ptr);
    const entries = try gpa.alloc(u32, n * k);
    defer gpa.free(entries);
    const got = try packBf12(rows, n, k, .{ .data = data, .row_base = bases, .row_ptr = row_ptr, .entries = entries, .n_entries = 0 });
    try std.testing.expectEqual(@as(usize, 0), got); // every value is inside its row's window
    for (0..n) |r| for (0..k) |c| {
        const tile = data[tileOffset(r / 8, c / 64, kg_n, total_n8)..][0..tile_bytes];
        const c64 = c % 64;
        const kt = c64 / 16;
        const d = c64 % 16;
        const jj: usize = (d / 8) * 2 + (d % 2);
        const lane = (r % 8) * 4 + (d % 8) / 2;
        const word = std.mem.readInt(u32, tile[(lane * 4 + kt) * 4 ..][0..4], .little);
        const sm_byte: u8 = @intCast((word >> ([4]u5{ 0, 16, 8, 24 })[jj]) & 0xFF);
        const ecw = std.mem.readInt(u32, tile[512 + lane * 8 + (kt >> 1) * 4 ..][0..4], .little);
        const sh: u5 = @intCast((kt & 1) * 8 + ([4]u8{ 0, 16, 4, 20 })[jj]);
        try std.testing.expectEqual(rows[r * k + c], decode(sm_byte, @intCast((ecw >> sh) & 0xF), bases[r]));
    };
}
