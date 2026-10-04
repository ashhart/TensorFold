//! Ordering: concurrent dispatch with memory barriers, and an indirect command buffer chain replayed with new arguments.
const std = @import("std");
const mtl = @import("metal");
const common = @import("common.zig");

const Gpu = common.Gpu;
const Size = mtl.Size;

fn mix(enc: mtl.ComputeEncoder, p: mtl.Pipeline, a: mtl.Buffer, b: mtl.Buffer, out: mtl.Buffer, n: u32) void {
    enc.setPipeline(p);
    enc.setBuffer(a, 0, 0);
    enc.setBuffer(b, 0, 1);
    enc.setBuffer(out, 0, 2);
    enc.setValue(n, 3);
    enc.dispatchThreads(Size.of(n, 1, 1), Size.of(256, 1, 1));
}

/// Wrong words in out[i] = A(in[i]) ^ (B(in[n - 1 - i]) * 2654435761), A and B tf_round maps.
fn wrong(out: []const u32, in: []const u32, a: [2]u32, b: [2]u32) usize {
    var bad: usize = 0;
    const n = out.len;
    for (out, 0..) |z, i| {
        const want = (a[0] *% in[i] +% a[1]) ^ ((b[0] *% in[n - 1 - i] +% b[1]) *% 2654435761);
        bad += @intFromBool(z != want);
    }
    return bad;
}

/// A = slow, B = fast, C = mix(A, B), A = fast, D = mix(A, B) with barriers between; without them mix reads A mid-write.
pub fn concurrentBarriers(gpu: Gpu) !void {
    const n: u32 = 1 << 20;
    const slow: u32 = 3000;
    const fast: u32 = 3;
    const p_round = try gpu.pipeline("tf_round", false);
    defer p_round.deinit();
    const p_mix = try gpu.pipeline("tf_mix", false);
    defer p_mix.deinit();
    var bufs: [5]mtl.Buffer = undefined;
    for (&bufs) |*b| b.* = try gpu.buffer(u32, n);
    defer for (bufs) |b| b.deinit();
    const in, const a, const b, const c, const d = bufs;
    const iv = in.slice(u32, n);
    var unordered: usize = 0;
    for (0..6) |round| {
        for (iv, 0..) |*v, i| v.* = @truncate(i *% 2246822519 +% round *% 3266489917);
        for ([_]bool{ true, false }) |barriers| {
            @memset(c.slice(u32, n), 0);
            @memset(d.slice(u32, n), 0);
            const cb = gpu.queue.commandBuffer();
            const enc = cb.compute(.concurrent);
            common.roundKernel(enc, p_round, in, 0, a, 0, n, slow);
            common.roundKernel(enc, p_round, in, 0, b, 0, n, fast);
            if (barriers) enc.barrier();
            mix(enc, p_mix, a, b, c, n);
            if (barriers) enc.barrier();
            common.roundKernel(enc, p_round, in, 0, a, 0, n, fast);
            if (barriers) enc.barrier();
            mix(enc, p_mix, a, b, d, n);
            enc.end();
            try common.run(cb);
            const bad = wrong(c.slice(u32, n), iv, common.lcgPower(slow), common.lcgPower(fast)) +
                wrong(d.slice(u32, n), iv, common.lcgPower(fast), common.lcgPower(fast));
            if (barriers and bad != 0) {
                std.debug.print("  round {d}: {d} wrong words with barriers\n", .{ round, bad });
                return error.Mismatch;
            }
            if (!barriers) unordered += bad;
        }
    }
    std.debug.print("  with barriers 0 wrong of {d} words; control without barriers: {d} wrong\n", .{ 12 * @as(usize, n), unordered });
}

const affine_count: u32 = 4096;

/// The chain y = ((x p0 + p1) p2 + p3) p4 + p5 on the host (exact for these small values).
fn chain(x: f32, p: []const f32) f32 {
    return ((x * p[0] + p[1]) * p[2] + p[3]) * p[4] + p[5];
}

fn replay(gpu: Gpu, icb: mtl.IndirectCommandBuffer, used: []const mtl.Buffer) !void {
    const cb = gpu.queue.commandBuffer();
    const enc = cb.compute(.concurrent);
    for (used) |buf| enc.useResource(buf, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
    enc.execute(icb, 0, icb.count);
    enc.end();
    try common.run(cb);
}

/// x -> t1 -> t2 -> out through three tf_affine commands; each waits for the one before (setBarrier).
pub fn icbReplay(gpu: Gpu) !void {
    const n = affine_count;
    const p = try gpu.pipeline("tf_affine", true);
    defer p.deinit();
    var bufs: [8]mtl.Buffer = undefined;
    for (&bufs) |*b| b.* = try gpu.buffer(f32, n);
    defer for (bufs) |b| b.deinit();
    const x0, const x1, const t1, const t2, const out0, const out1, const params, const count = bufs;
    for (x0.slice(f32, n), x1.slice(f32, n), 0..) |*u, *v, i| {
        u.* = @floatFromInt(i % 97);
        v.* = @floatFromInt((i * 7) % 89);
    }
    const pv = params.slice(f32, 6);
    @memcpy(pv, &[_]f32{ 2, 1, 3, -4, -1, 5 });
    count.slice(u32, 1)[0] = n;

    const icb = try mtl.IndirectCommandBuffer.init(gpu.device, 3, 4);
    defer icb.deinit();
    const links = [_][2]mtl.Buffer{ .{ x0, t1 }, .{ t1, t2 }, .{ t2, out0 } };
    for (links, 0..) |link, k| {
        const cmd = icb.command(k);
        cmd.setPipeline(p);
        cmd.setBuffer(link[0], 0, 0);
        cmd.setBuffer(link[1], 0, 1);
        cmd.setBuffer(params, 8 * k, 2);
        cmd.setBuffer(count, 0, 3);
        cmd.dispatchThreads(Size.of(n, 1, 1), Size.of(64, 1, 1));
        cmd.setBarrier();
    }
    const used = bufs[0..];

    // replay 1: as encoded
    try replay(gpu, icb, used);
    for (out0.slice(f32, n), x0.slice(f32, n)) |y, x| if (y != chain(x, pv)) return error.Replay1;

    // replay 2: new parameter values, input rebound to x1, output rebound to out1
    @memcpy(pv, &[_]f32{ -2, 3, 0.5, 1, 4, -8 });
    icb.command(0).setBuffer(x1, 0, 0);
    icb.command(2).setBuffer(out1, 0, 1);
    try replay(gpu, icb, used);
    for (out1.slice(f32, n), x1.slice(f32, n)) |y, x| if (y != chain(x, pv)) return error.Replay2;
    for (out0.slice(f32, n), x0.slice(f32, n)) |y, x| if (y != chain(x, &[_]f32{ 2, 1, 3, -4, -1, 5 })) return error.Replay2Clobbered;

    // replay 3: input at a byte offset (x0 from element 16) over n - 16 elements
    @memcpy(pv, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    icb.command(0).setBuffer(x0, 16 * @sizeOf(f32), 0);
    count.slice(u32, 1)[0] = n - 16;
    try replay(gpu, icb, used);
    const y = out1.slice(f32, n);
    const x = x0.slice(f32, n);
    for (0..n - 16) |i| if (y[i] != chain(x[i + 16], pv)) return error.Replay3;
    for (n - 16..n) |i| if (y[i] != chain(x1.slice(f32, n)[i], &[_]f32{ -2, 3, 0.5, 1, 4, -8 })) return error.Replay3Tail;
}
