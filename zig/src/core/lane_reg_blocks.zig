//! Pure host rewrites from tiled quantization words to tensor-op register blocks.

/// The 4-bit register layout for one 32-column x 64-code block.
pub fn block(tiled: *const [256]u32, out: *[256]u32) void {
    for (out, 0..) |*o, t| {
        const lane = (t >> 2) & 31;
        const w = 4 * (t >> 7) + (t & 3);
        const at = (((lane >> 1) & 3) + 4 * ((lane >> 4) & 1) + 8 * (w >> 1)) * 8 + ((lane >> 3) & 1) + 4 * (w & 1);
        const shift: u5 = @intCast(16 * (lane & 1));
        const a = (tiled[at] >> shift) & 0xffff;
        const c = (tiled[at + 2] >> shift) & 0xffff;
        o.* = (a & 0xf) | ((a >> 4) & 0xf0) | ((c & 0xf) << 8) | ((c << 4) & 0xf000) | (((a >> 4) & 0xf) << 16) | (((a >> 8) & 0xf0) << 16) | (((c >> 4) & 0xf) << 24) | (((c >> 8) & 0xf0) << 24);
    }
}

/// The split-plane register layout for one 5- or 6-bit 32x64 block.
pub fn split(comptime bits: u8, tiled: []const u32, out: []u32) void {
    const hi_bits = bits - 4;
    const row_words = 2 * bits;
    for (0..256) |t| {
        const lane = (t >> 2) & 31;
        const col = coordCol(lane, 4 * (t >> 7) + (t & 3));
        var lo4: u32 = 0;
        for (0..8) |j| lo4 |= (tileCode(tiled, row_words, col, coordK(lane, 4 * (t >> 7) + (t & 3), j), bits) & 15) << @intCast(4 * j);
        out[t] = lo4;
    }
    const slots = 4 / hi_bits;
    const mask = (@as(u32, 1) << hi_bits) - 1;
    for (0..64 * hi_bits) |t| {
        const hl = t >> hi_bits;
        const p = t & ((1 << hi_bits) - 1);
        var hi_w: u32 = 0;
        for (0..slots) |s| {
            const hw = slots * p + s;
            const hcol = coordCol(hl, hw);
            for (0..4) |j| {
                const c0 = (tileCode(tiled, row_words, hcol, coordK(hl, hw, j), bits) >> 4) & mask;
                const c1 = (tileCode(tiled, row_words, hcol, coordK(hl, hw, j + 4), bits) >> 4) & mask;
                const sh: u5 = @intCast(s * 4 * hi_bits + j * hi_bits);
                hi_w |= (c0 << sh) | (c1 << (@as(u5, 16) + sh));
            }
        }
        out[256 + t] = hi_w;
    }
}

fn coordCol(lane: usize, w: usize) usize {
    return ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1) + 8 * (w >> 1);
}

fn coordK(lane: usize, w: usize, j: usize) usize {
    const k0 = 4 * (lane & 1) + 8 * ((lane >> 3) & 1) + 32 * (w & 1);
    return k0 + (j & 1) * 2 + ((j >> 1) & 1) * 16 + (j >> 2);
}

fn tileCode(tiled: []const u32, row_words: usize, col: usize, k: usize, comptime bits: u8) u32 {
    const bit = k * bits;
    const idx = col * row_words + (bit >> 5);
    const sh: u5 = @intCast(bit & 31);
    var v = tiled[idx] >> sh;
    if (@as(usize, sh) + bits > 32) v |= tiled[idx + 1] << @intCast(32 - @as(usize, sh));
    return v & ((@as(u32, 1) << @intCast(bits)) - 1);
}
