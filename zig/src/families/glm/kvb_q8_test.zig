//! Synthetic GPU parity tests for GLM-5.3-Flash intact 8/64 kv_b absorb, absorb_nax_8b (TP1/TP2), and unabsorb.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const wts = @import("weights.zig");
const kernels = @import("kernels.zig");

const bf16_round = @import("bf16_round");
const bf16 = bf16_round.bits;
fn f32of(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}
fn ref(b: mtl.Buffer) wts.Ref {
    return .{ .buf = b };
}

fn pipeFromSource(a: std.mem.Allocator, device: mtl.Device, raw: []const u8, sym: [:0]const u8) !mtl.Pipeline {
    const lib = try mtl.Library.fromSource(device, raw, mtl.CompileOptions.mlx());
    defer lib.deinit();
    _ = a;
    return mtl.Pipeline.init(device, lib, sym, false);
}

fn checkAbsorbCase(
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    k: *const kernels.Kernels,
    nax_pipe: mtl.Pipeline,
    H: usize,
) !void {
    const NOPE: usize = 256;
    const LATENT: usize = 512;
    const PER_HEAD: usize = 512;
    const M: usize = 70;
    const QS: usize = H * NOPE + 64;
    const WR = H * PER_HEAD;
    const opts = mtl.ResourceOptions.shared;
    const bufs = [_]mtl.Buffer{
        try device.buffer(WR * LATENT, opts),
        try device.buffer(WR * (LATENT / 64) * 2, opts),
        try device.buffer(WR * (LATENT / 64) * 2, opts),
        try device.buffer(M * QS * 2, opts),
        try device.buffer(M * H * LATENT * 2, opts),
        try device.buffer(M * H * LATENT * 2, opts),
        try device.buffer(3 * H * LATENT * 2, opts),
        try device.buffer(3 * H * NOPE * 2, opts),
    };
    defer for (bufs) |b| b.deinit();
    var prng = std.Random.DefaultPrng.init(0xab08 ^ @as(u64, H));
    const rnd = prng.random();
    const wb = bufs[0].slice(u8, WR * LATENT);
    rnd.bytes(wb);
    const sc = bufs[1].slice(u16, WR * (LATENT / 64));
    const bi = bufs[2].slice(u16, WR * (LATENT / 64));
    for (sc, bi) |*s, *b| {
        s.* = bf16((rnd.float(f32) + 0.5) / 512.0);
        b.* = bf16(-(rnd.float(f32) + 0.5) * 0.25);
    }
    const qp = bufs[3].slice(u16, M * QS);
    for (qp) |*v| v.* = bf16(rnd.float(f32) * 4 - 2);
    const att = bufs[6].slice(u16, 3 * H * LATENT);
    for (att) |*v| v.* = bf16(rnd.float(f32) * 4 - 2);
    const q: wts.Affine = .{ .w = ref(bufs[0]), .s = ref(bufs[1]), .b = ref(bufs[2]), .n = @intCast(WR), .k = LATENT, .bits = 8, .group = 64 };
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    var at_row: u32 = 0;
    while (at_row < M) : (at_row += 16) {
        const chunk = @min(@as(u32, 16), @as(u32, @intCast(M)) - at_row);
        kernels.absorb8(k, enc, q, .{ .buf = bufs[3], .off = at_row * QS * 2 }, .{ .buf = bufs[4], .off = at_row * H * LATENT * 2 }, @intCast(QS), @intCast(H), chunk, NOPE, LATENT, PER_HEAD);
    }
    enc.setPipeline(nax_pipe);
    for (bufs[0..4], 0..) |b, i| enc.setBuffer(b, 0, i);
    enc.setBuffer(bufs[5], 0, 4);
    enc.setValue([2]i32{ M, @intCast(QS) }, 5);
    enc.dispatchGroups(mtl.Size.of(LATENT / 64, (M + 63) / 64, H), mtl.Size.of(128, 1, 1));
    if (H == 64) kernels.unabsorb8(k, enc, q, ref(bufs[6]), ref(bufs[7]), @intCast(H), 3, LATENT, NOPE, PER_HEAD, NOPE);
    enc.end();
    cb.commit();
    cb.wait();
    try std.testing.expect(cb.failure() == null);
    const dec_out = bufs[4].slice(u16, M * H * LATENT);
    const nax_out = bufs[5].slice(u16, M * H * LATENT);
    var worst: f64 = 0;
    for (dec_out, nax_out) |dv, nv| {
        const allow = @abs(@as(f64, f32of(dv))) / 64.0 + 1e-4;
        worst = @max(worst, @abs(@as(f64, f32of(nv)) - @as(f64, f32of(dv))) / allow);
    }
    try std.testing.expect(worst <= 1.0);
    const rows_chk = [_]usize{ 0, 15, 16, 63, 64, 69 };
    const heads_chk: []const usize = if (H == 64) &.{ 0, 1, 31, 63 } else &.{ 0, 15, 31 };
    for (rows_chk) |r| for (heads_chk) |h| for (0..LATENT) |j| {
        var o: f64 = 0;
        for (0..NOPE) |i| {
            const wrow = h * PER_HEAD + i;
            const g = wrow * (LATENT / 64) + j / 64;
            const w: f64 = @as(f32, f32of(sc[g]) * @as(f32, @floatFromInt(wb[wrow * LATENT + j])) + f32of(bi[g]));
            o += w * f32of(qp[r * QS + h * NOPE + i]);
        }
        const idx = (r * H + h) * LATENT + j;
        const allow = @abs(o) / 64.0 + 1e-4;
        try std.testing.expect(@abs(@as(f64, f32of(dec_out[idx])) - o) <= allow);
        try std.testing.expect(@abs(@as(f64, f32of(nax_out[idx])) - o) <= allow);
    };
    if (H == 64) {
        const vals_out = bufs[7].slice(u16, 3 * H * NOPE);
        for (0..3) |r| for (heads_chk) |h| for (0..NOPE) |i| {
            var o: f64 = 0;
            const wrow = h * PER_HEAD + NOPE + i;
            for (0..LATENT) |j| {
                const g = wrow * (LATENT / 64) + j / 64;
                const w: f64 = @as(f32, f32of(sc[g]) * @as(f32, @floatFromInt(wb[wrow * LATENT + j])) + f32of(bi[g]));
                o += w * f32of(att[(r * H + h) * LATENT + j]);
            }
            try std.testing.expect(@abs(@as(f64, f32of(vals_out[(r * H + h) * NOPE + i])) - o) <= @abs(o) / 64.0 + 1e-4);
        };
    }
    _ = a;
}

test "8/64 intact kv_b absorb_nax_8b (TP1/TP2), absorb8, and unabsorb8 match float64 within bf16 rounding" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    var k: kernels.Kernels = undefined;
    k.absorb_8b = try pipeFromSource(a, device, sources.glm_kvb_q8, "glm_absorb_8b");
    defer k.absorb_8b.deinit();
    k.unabsorb_8b = try pipeFromSource(a, device, sources.glm_kvb_q8, "glm_unabsorb_8b");
    defer k.unabsorb_8b.deinit();
    const raw1 = try std.mem.concat(a, u8, &.{ "#define GLM_KV_BITS 8\n", sources.glm_absorb_nax });
    defer a.free(raw1);
    const src1 = try frags.source(device, a, raw1);
    defer a.free(src1);
    const nax1 = try pipeFromSource(a, device, src1, "glm_absorb_nax");
    defer nax1.deinit();
    const raw2 = try std.mem.concat(a, u8, &.{ "#define GLM_HEADS 32\n#define GLM_KV_BITS 8\n", sources.glm_absorb_nax });
    defer a.free(raw2);
    const src2 = try frags.source(device, a, raw2);
    defer a.free(src2);
    const nax2 = try pipeFromSource(a, device, src2, "glm_absorb_nax");
    defer nax2.deinit();
    try checkAbsorbCase(a, device, queue, &k, nax1, 64);
    try checkAbsorbCase(a, device, queue, &k, nax2, 32);
}
