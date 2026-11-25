//! NVFP4_NZ5's codes in contract_nvfp4_fact.zig's fills, and the float64 weight its block 32 entries read.
const std = @import("std");
const core = @import("core");

/// No 0b1000 in E2M1's codes, nz their copy; planted groups hold +5s alone in nz and 4s under scales x1.25 in codes.
pub fn plant(codes: []u8, nz: []u8, scales: []u8, rows: usize, k: usize, rng: std.Random, planted: bool) void {
    const g16 = k / 16;
    for (codes) |*b| {
        if (b.* & 0x0f == 8) b.* &= 0xf0;
        if (b.* >> 4 == 8) b.* &= 0x0f;
    }
    @memcpy(nz, codes);
    if (!planted) return;
    for (0..rows) |r| for (0..k / 32) |blk| {
        const s = scales[r * g16 + 2 * blk];
        if (s & 7 != 0 or s + 2 > 0x7e or rng.uintLessThan(u8, 4) != 0) continue; // a quarter of the powers of two
        const at = r * k / 2 + blk * 16;
        @memset(codes[at..][0..16], 0);
        @memset(nz[at..][0..16], 0);
        for (0..1 + rng.uintLessThan(usize, 3)) |_| {
            const i = rng.uintLessThan(usize, 32);
            put(codes[at..], i, 6); // e2m1's 4
            put(nz[at..], i, 8);
        }
        @memset(scales[r * g16 + 2 * blk ..][0..2], s + 2);
    };
}

/// Gate 4's +5s among the stress fill's 4s and 6s: two on equal inputs against a -4 and a -6, as there is no -5.
pub fn stress(x: []u16, nz: []u8, x_rows: usize, rows: usize, k: usize, rng: std.Random) void {
    const h = k / 2;
    for (0..x_rows) |r| for (0..h / 32) |gi| {
        const i = r * k + fiveAt(gi);
        x[i + 1] = x[i];
        x[h + i + 1] = x[h + i];
    };
    for (0..rows) |j| for (0..h / 32) |gi| {
        const line = nz[j * k / 2 ..];
        const i = fiveAt(gi);
        switch (rng.uintLessThan(u8, 3)) {
            0 => {}, // the stress fill's own mirror
            1 => for ([_]usize{ i, i + 1, h + i, h + i + 1 }, [_]u8{ 8, 8, 0xe, 0xf }) |at, v| put(line, at, v), // 5x + 5x - 4x - 6x
            else => for ([_]usize{ i, i + 1, h + i, h + i + 1 }, [_]u8{ 0xe, 0xf, 8, 8 }) |at, v| put(line, at, v), // the halves swapped
        }
    };
}

/// Group gi's pair of +5 inputs in K's first half: an even offset from 2 to 30, never input 0, which is unmirrored.
fn fiveAt(gi: usize) usize {
    return gi * 32 + 2 + 2 * (gi % 15);
}

/// Row r's input i as NVFP4_NZ5's entries read it: nz's nibble, 0b1000 as +5, under its group of 32's e4m3 scale.
pub fn weight(nz: []const u8, bases: []const u8, offsets: []const u8, k: usize, r: usize, i: usize) f64 {
    const b = nz[r * k / 2 + i / 2];
    const nib: u4 = @truncate(if (i % 2 == 0) b & 0xf else b >> 4);
    const grp = i / 32;
    const off: u8 = (offsets[r * (k / 64) + grp / 2] >> if (grp % 2 == 0) @as(u3, 0) else 4) & 0xf;
    return (if (nib == 8) 5 else core.quant.nvfp4.e2m1(nib)) * core.quant.fp8.e4m3(bases[r] + off);
}

fn put(codes: []u8, i: usize, v: u8) void {
    const b = &codes[i / 2];
    b.* = if (i % 2 == 0) (b.* & 0xf0) | v else (b.* & 0x0f) | v << 4;
}
