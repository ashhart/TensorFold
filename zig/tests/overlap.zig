//! Host/GPU overlap: the host prepares round k + 1 while the GPU runs round k, event-gated or committed ahead.
const std = @import("std");
const mtl = @import("metal");
const common = @import("common.zig");

const Gpu = common.Gpu;
const n: u32 = 1 << 16;
const rounds = 48;
const iters: u32 = 4000;

fn input(round: usize, i: usize) u32 {
    return @truncate((round + 1) *% 2654435761 +% i *% 40503);
}

/// Host work for one round: write its input, then spin until `busy` seconds have passed.
fn prepare(slots: []u32, round: usize, busy: f64) void {
    const start = mtl.clock.seconds();
    for (slots[(round % 2) * n ..][0..n], 0..) |*v, i| v.* = input(round, i);
    while (mtl.clock.seconds() - start < busy) std.atomic.spinLoopHint();
}

fn encode(cb: mtl.CommandBuffer, p: mtl.Pipeline, in: mtl.Buffer, out: mtl.Buffer, round: usize) void {
    const enc = cb.compute(.serial);
    common.roundKernel(enc, p, in, (round % 2) * n * @sizeOf(u32), out, round * n * @sizeOf(u32), n, iters);
    enc.end();
}

fn verify(out: mtl.Buffer) !void {
    const map = common.lcgPower(iters);
    const all = out.slice(u32, rounds * n);
    for (0..rounds) |r| {
        for (all[r * n .. (r + 1) * n], 0..) |v, i| if (v != map[0] *% input(r, i) +% map[1]) {
            std.debug.print("  round {d} word {d}: {d}, want {d}\n", .{ r, i, v, map[0] *% input(r, i) +% map[1] });
            return error.Mismatch;
        };
    }
    @memset(all, 0);
}

/// Median GPU idle time between consecutive rounds, in microseconds.
fn medianGap(cbs: []const mtl.CommandBuffer) f64 {
    var gaps: [rounds - 1]f64 = undefined;
    for (&gaps, 1..) |*g, r| g.* = (cbs[r].gpuStart() - cbs[r - 1].gpuEnd()) * 1e6;
    return common.median(&gaps);
}

pub fn eventOverlap(gpu: Gpu) !void {
    const p = try gpu.pipeline("tf_round", false);
    defer p.deinit();
    const in = try gpu.buffer(u32, 2 * n);
    defer in.deinit();
    const out = try gpu.buffer(u32, rounds * n);
    defer out.deinit();
    const slots = in.slice(u32, 2 * n);
    const done = try gpu.device.sharedEvent();
    defer done.deinit();
    const host = try gpu.device.sharedEvent();
    defer host.deinit();

    // calibrate: one round's GPU time; the host's work per round is made the same length
    var busy: f64 = 0;
    for (0..5) |k| {
        const cb = gpu.queue.commandBuffer();
        encode(cb, p, in, out, 0);
        try common.run(cb);
        if (k > 0) busy = @max(busy, cb.gpuSeconds());
    }

    // serial: prepare, submit, wait
    const t0 = mtl.clock.seconds();
    for (0..rounds) |r| {
        prepare(slots, r, busy);
        const cb = gpu.queue.commandBuffer();
        encode(cb, p, in, out, r);
        try common.run(cb);
    }
    const serial = mtl.clock.seconds() - t0;
    try verify(out);

    // event-gated: every round committed up front, waiting for its host signal; `done` counts finished rounds
    var gated: [rounds]mtl.CommandBuffer = undefined;
    var signaled: [rounds]f64 = undefined;
    const t1 = mtl.clock.seconds();
    for (&gated, 0..) |*cb, r| {
        cb.* = gpu.queue.commandBuffer();
        cb.waitFor(host, r + 1);
        encode(cb.*, p, in, out, r);
        cb.signal(done, r + 1);
        cb.commit();
    }
    for (0..rounds) |r| {
        if (r >= 2 and !done.wait(r - 1, 5000)) return error.Timeout; // round r - 2 has read this slot
        prepare(slots, r, busy);
        signaled[r] = mtl.clock.seconds();
        host.set(r + 1);
    }
    if (!done.wait(rounds, 5000)) return error.Timeout;
    const event_time = mtl.clock.seconds() - t1;
    for (gated) |cb| if (cb.failure() != null) return error.GpuFailed;
    try verify(out);
    // wake latency: host signal to GPU start, over rounds whose GPU was idle when signaled
    var wake: [rounds]f64 = undefined;
    var idle: usize = 0;
    for (0..rounds) |r| {
        if (r > 0 and signaled[r] < gated[r - 1].gpuEnd()) continue;
        wake[idle] = (gated[r].gpuStart() - signaled[r]) * 1e6;
        idle += 1;
    }

    // commit-ahead: no host signal on the GPU's path; round r is committed as soon as its input is written
    var ahead: [rounds]mtl.CommandBuffer = undefined;
    const base = done.value();
    const t2 = mtl.clock.seconds();
    for (&ahead, 0..) |*cb, r| {
        if (r >= 2 and !done.wait(base + r - 1, 5000)) return error.Timeout;
        prepare(slots, r, busy);
        cb.* = gpu.queue.commandBuffer();
        encode(cb.*, p, in, out, r);
        cb.signal(done, base + r + 1);
        cb.commit();
    }
    if (!done.wait(base + rounds, 5000)) return error.Timeout;
    const ahead_time = mtl.clock.seconds() - t2;
    for (ahead) |cb| if (cb.failure() != null) return error.GpuFailed;
    try verify(out);

    std.debug.print("  {d} rounds of GPU {d:.0} us + host {d:.0} us: serial {d:.1} ms; event-gated {d:.1} ms ({d:.2}x, GPU idle {d:.1} us a round, wake {d:.1} us median of {d}); commit-ahead {d:.1} ms ({d:.2}x, GPU idle {d:.1} us a round)\n", .{
        rounds,                       busy * 1e6, busy * 1e6,       serial * 1e3,        event_time * 1e3,  serial / event_time, medianGap(&gated),
        common.median(wake[0..idle]), idle,       ahead_time * 1e3, serial / ahead_time, medianGap(&ahead),
    });
    if (event_time > 0.8 * serial or ahead_time > 0.8 * serial) return error.NoOverlap;
}
