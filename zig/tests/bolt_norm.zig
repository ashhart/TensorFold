//! Bolt's add_norm_moe_xs instantiation on the GPU (its gemv, route and combine: tools/zig/bolt_op_cases.py vs MLX).
const std = @import("std");
const mtl = @import("metal");
const common = @import("common.zig");

const Gpu = common.Gpu;
const Size = mtl.Size;

fn bf16(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

fn f32of(w: u16) f32 {
    return @bitCast(@as(u32, w) << 16);
}

/// add_norm_moe_xs_bolt <4096, 512, 1> with no picks and no shared expert: OUT = rmsnorm(H) * W, HN = H, XS its sums.
pub fn addNormBolt(gpu: Gpu) !void {
    const D: usize = 4096; // Bolt hidden, baked into the instantiation
    const E: usize = 1; // Bolt's MoE output is combined and projected before add_norm: one pick
    const rows: usize = 2;
    const mp: usize = 16; // rows padded to 16, the XS stride
    const eps: f32 = 1e-5;

    const h = try gpu.buffer(u16, rows * D);
    defer h.deinit();
    const yexp = try gpu.buffer(u16, rows * E * D);
    defer yexp.deinit();
    const we = try gpu.buffer(f32, rows * E);
    defer we.deinit();
    const sh = try gpu.buffer(u16, rows * D);
    defer sh.deinit();
    const wnorm = try gpu.buffer(u16, D);
    defer wnorm.deinit();
    const hn = try gpu.buffer(u16, rows * D);
    defer hn.deinit();
    const out = try gpu.buffer(u16, rows * D);
    defer out.deinit();
    const xs = try gpu.buffer(f32, D / 64 * mp);
    defer xs.deinit();

    var prng = std.Random.DefaultPrng.init(0xadd_0_117);
    const rnd = prng.random();
    const hv = h.slice(u16, rows * D);
    for (hv) |*v| v.* = bf16(rnd.float(f32) - 0.5);
    @memset(yexp.slice(u16, rows * E * D), 0); // no routed contribution
    @memset(we.slice(f32, rows * E), 0);
    @memset(sh.slice(u16, rows * D), 0); // no shared contribution
    for (wnorm.slice(u16, D)) |*v| v.* = bf16(1.0); // W = 1, so OUT is the pure rmsnorm
    @memset(hn.slice(u16, rows * D), 0xffff);
    @memset(out.slice(u16, rows * D), 0xffff);

    const p = try gpu.pipeline("tf_nemotron_add_norm_moe_xs_bolt", false);
    defer p.deinit();
    const cb = gpu.queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(p);
    enc.setBuffer(h, 0, 0);
    enc.setBuffer(yexp, 0, 1);
    enc.setBuffer(we, 0, 2);
    enc.setBuffer(sh, 0, 3);
    enc.setBuffer(wnorm, 0, 4);
    enc.setValue(eps, 5);
    enc.setValue([8]i32{ @intCast(rows), @intCast(mp), 0, 0, 0, 0, 0, 0 }, 6);
    enc.setBuffer(hn, 0, 7);
    enc.setBuffer(out, 0, 8);
    enc.setBuffer(xs, 0, 9);
    enc.dispatchThreads(Size.of(512 * mp, 1, 1), Size.of(512, 1, 1));
    enc.end();
    try common.run(cb);

    const got_hn = hn.slice(u16, rows * D);
    const got_out = out.slice(u16, rows * D);
    const got_xs = xs.slice(f32, D / 64 * mp);
    for (0..rows) |r| {
        var ss: f32 = 0;
        for (0..D) |c| ss += f32of(hv[r * D + c]) * f32of(hv[r * D + c]);
        const scale = 1.0 / @sqrt(ss / @as(f32, D) + eps);
        for (0..D) |c| {
            if (got_hn[r * D + c] != hv[r * D + c]) return error.HnChanged; // delta is 0, so HN == H
            const want = f32of(hv[r * D + c]) * scale;
            if (@abs(f32of(got_out[r * D + c]) - want) > 0.02 * @abs(want) + 1e-2) {
                std.debug.print("  add_norm_bolt out[{d},{d}]: got {d}, reference {d}\n", .{ r, c, f32of(got_out[r * D + c]), want });
                return error.WrongNorm;
            }
        }
        // XS[g] is the sum of OUT over each 64-wide group
        for (0..D / 64) |g| {
            var acc: f32 = 0;
            for (0..64) |i| acc += f32of(got_out[r * D + g * 64 + i]);
            if (@abs(got_xs[g * mp + r] - acc) > 0.05 * @abs(acc) + 1e-2) return error.WrongSums;
        }
    }
}
