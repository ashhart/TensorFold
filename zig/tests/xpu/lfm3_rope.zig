//! LFM3 per-head QK RMSNorm with zero-tail partial RoPE, and the V store, against FP64 on random data.

const std = @import("std");
const o = @import("lfm3_ops.zig");
const stop = @import("xpu").stop;

const HD = 128;
const eps: f64 = 1e-5;

/// Inverse frequencies as upstream builds them in fp32: base^(-2i/128) for i < angles, then zeros.
fn invFreq(base: f32, angles: usize, out: []f32) void {
    for (out, 0..) |*f, i| f.* = if (i < angles) 1.0 / std.math.pow(f32, base, @as(f32, @floatFromInt(2 * i)) / 128.0) else 0;
}

/// FP64 replay of one row's heads: RMSNorm over 128 dims, then rotate_half RoPE at `pos` with `angles` rotated pairs.
fn refRow(src: []const f64, w: []const f64, heads: usize, base: f64, angles: usize, pos: f64, out: []f64) void {
    for (0..heads) |h| {
        const s = src[h * HD ..][0..HD];
        var ss: f64 = 0;
        for (s) |v| ss += v * v;
        const rs = 1.0 / @sqrt(ss / HD + eps);
        for (0..64) |i| {
            const a = w[i] * s[i] * rs;
            const b = w[i + 64] * s[i + 64] * rs;
            const f = if (i < angles) pos / std.math.pow(f64, base, @as(f64, @floatFromInt(2 * i)) / 128.0) else 0;
            out[h * HD + i] = a * @cos(f) - b * @sin(f);
            out[h * HD + i + 64] = b * @cos(f) + a * @sin(f);
        }
    }
}

const Case = struct { what: []const u8, base: f32, angles: usize, heads: u32, pos0: u32, ring: u32 };

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    stop.install();
    var r = try o.rt.open();
    defer r.deinit();
    var m = try r.module(@import("xpu").kernels.lfm3);
    var kr = try m.kernel("lfm3_qk_norm_rope", .{ 64, 1, 1 });
    var kv = try m.kernel("lfm3_v_store", .{ 64, 1, 1 });
    var t: o.Tally = .{ .name = "lfm3_rope" };
    var prng = std.Random.DefaultPrng.init(12);
    const rnd = prng.random();

    const n: u32 = 9;
    const stride: u32 = 2048 + 2 * 512; // a fused q | k | v projection row
    const sh = try o.gpa.alloc(u16, n * stride);
    o.randBf(sh, rnd, 1.5);
    const src = try o.upload(&r, sh);
    const sf = try o.widen(o.gpa, sh);
    const wh = try o.gpa.alloc(u16, HD);
    for (wh) |*v| v.* = o.toBf(1.0 + 0.3 * rnd.floatNorm(f64));
    const w = try o.upload(&r, wh);
    const wf = try o.widen(o.gpa, wh);
    const inv = try r.alloc(64 * 4);
    var invh: [64]f32 = undefined;
    std.debug.print("QK RMSNorm + RoPE over 128-dim heads, {d} rows, against FP64\n", .{n});

    const cases = [_]Case{
        .{ .what = "q, full layer (theta 1e6, 32 of 64 pairs)", .base = 1e6, .angles = 32, .heads = 16, .pos0 = 7000, .ring = 0 },
        .{ .what = "q, sliding layer (theta 1e4, all 64 pairs)", .base = 1e4, .angles = 64, .heads = 16, .pos0 = 5, .ring = 0 },
        .{ .what = "k into a 256-slot ring across its wrap", .base = 1e4, .angles = 64, .heads = 4, .pos0 = 250, .ring = 256 },
    };
    var full_out: []u16 = &.{};
    for (cases) |c| {
        invFreq(c.base, c.angles, &invh);
        try r.upload(inv, std.mem.sliceAsBytes(&invh));
        const slots: u32 = if (c.ring == 0) n else c.ring;
        const dst_stride: u32 = c.heads * HD;
        const dst = try r.alloc(slots * dst_stride * 2);
        const src_off: u32 = if (c.heads == 16) 0 else 2048;
        try o.run(&kr, .{ c.heads, n, 1 }, .{ src, stride, src_off, dst, dst_stride, w, inv, c.pos0, c.ring, @as(f32, 1e-5) });
        const dh = try o.fetch(&r, dst, slots * dst_stride);
        const got = try o.gpa.alloc(u16, n * dst_stride);
        const want = try o.gpa.alloc(f64, n * dst_stride);
        for (0..n) |z| {
            const slot = if (c.ring == 0) z else (c.pos0 + z) % c.ring;
            @memcpy(got[z * dst_stride ..][0..dst_stride], dh[slot * dst_stride ..][0..dst_stride]);
            refRow(sf[z * stride + src_off ..], wf, c.heads, c.base, c.angles, @floatFromInt(c.pos0 + z), want[z * dst_stride ..]);
        }
        t.check(c.what, o.relErr(got, want));
        if (c.angles == 32) full_out = got;
    }

    // V rows into the ring: a plain copy
    {
        const ring: u32 = 256;
        const pos0: u32 = 250;
        const dst = try r.alloc(ring * 512 * 2);
        try o.run(&kv, .{ 512 / 64, n, 1 }, .{ src, stride, @as(u32, 2048 + 512), dst, @as(u32, 512), pos0, ring });
        const dh = try o.fetch(&r, dst, ring * 512);
        var eq = true;
        for (0..n) |z| {
            const slot = (pos0 + z) % ring;
            if (!std.mem.eql(u16, dh[slot * 512 ..][0..512], sh[z * stride + 2048 + 512 ..][0..512])) eq = false;
        }
        t.same("v rows stored at slot (pos0 + r) % 256", eq);
    }

    // planted control: the full layer's q replayed with every pair rotated (partial rotary ignored) must not match
    {
        const want = try o.gpa.alloc(f64, n * 16 * HD);
        for (0..n) |z| refRow(sf[z * stride ..], wf, 16, 1e6, 64, @floatFromInt(7000 + z), want[z * 16 * HD ..]);
        t.control("full layer with all 64 pairs rotated", o.relErr(full_out, want));
    }
    try t.finish();
}
