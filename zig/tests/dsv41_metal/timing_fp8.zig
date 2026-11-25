//! Each FP8 projection of a synthetic release-shape layer alone: the row kernel against the MMA tile at 1-16 rows.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const fixture = @import("fixture.zig");
const timing = @import("timing_real.zig");

const counters = mtl.counters;
const fp8 = pkg.pipelines.fp8;
const tile = pkg.pipelines.fp8_tile;

const Arm = union(enum) { rows, tile: tile.Form };

fn encode(r: *fixture.Rig, e: mtl.ComputeEncoder, w: pkg.weights.Fp8, x: mtl.Buffer, y: mtl.Buffer, rows: u32, arm: Arm) !void {
    switch (arm) {
        .rows => {
            const call: fp8.Call = .{ .k = w.k, .n = w.n, .rows = rows, .split = fp8.splitOf(w.k), .out = .bf16 };
            try fp8.encodeWhole(&r.p.fp8, e, .{ .buf = x }, .{ .buf = w.w.buf, .off = w.w.off }, .{ .buf = w.s.buf, .off = w.s.off }, .{ .buf = y }, call, .{});
        },
        .tile => |f| {
            const call: tile.Call = .{ .k = w.k, .n = w.n, .rows = rows, .out = .bf16 };
            try tile.encodeAs(&r.p.tile, e, .{ .buf = x }, .{ .buf = w.w.buf, .off = w.w.off }, .{ .buf = w.s.buf, .off = w.s.off }, .{ .buf = y }, call, .{}, f);
        },
    }
}

fn once(r: *fixture.Rig, junk: mtl.Buffer, samples: counters.SampleBuffer, pass: counters.ComputePass, w: pkg.weights.Fp8, x: mtl.Buffer, y: mtl.Buffer, rows: u32, arm: Arm) !f64 {
    timing.flush(r, junk);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const before = counters.sampleClocks(r.device);
    const cb = r.queue.commandBuffer();
    const e = try pass.compute(cb, samples, 0, 1);
    try encode(r, e, w, x, y, rows, arm);
    e.end();
    cb.commit();
    cb.wait();
    const res = try samples.resolveCompleted(2);
    defer res.deinit();
    const after = counters.sampleClocks(r.device);
    const cpu: f64 = @floatFromInt(after.cpu - before.cpu);
    const gpu: f64 = @floatFromInt(after.gpu - before.gpu);
    const v = res.values();
    return @as(f64, @floatFromInt(v[1].timestamp -% v[0].timestamp)) * cpu / gpu * 1e-9;
}

/// Fastest of `n` cold launches of each arm, interleaved, and whether every tile arm wrote the row kernel's bytes.
pub fn run(r: *fixture.Rig, junk: mtl.Buffer, n: usize) !void {
    const l = &r.layers[1];
    const Shape = struct { name: []const u8, w: pkg.weights.Fp8 };
    const shapes = [_]Shape{ .{ .name = "wq_a", .w = l.wq_a }, .{ .name = "wkv", .w = l.wkv }, .{ .name = "wq_b", .w = l.wq_b }, .{ .name = "wo_b", .w = l.wo_b }, .{ .name = "shared w1", .w = l.shared[0] }, .{ .name = "shared w2", .w = l.shared[1] }, .{ .name = "ix_wq_b", .w = l.ix_wq_b.? } };
    const x = try r.device.buffer(16 * 8192 * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    var prng = std.Random.DefaultPrng.init(3);
    for (x.slice(u16, 16 * 8192)) |*v| v.* = @as(u16, @intCast(0x3c00 + prng.random().uintLessThan(u16, 0x0400))) ^ (@as(u16, prng.random().int(u1)) << 15);
    const ya = try r.device.buffer(16 * 32768 * 2, mtl.ResourceOptions.shared);
    defer ya.deinit();
    const yb = try r.device.buffer(16 * 32768 * 2, mtl.ResourceOptions.shared);
    defer yb.deinit();
    const samples = try counters.SampleBuffer.init(r.device, 64);
    defer samples.deinit();
    const pass = try counters.ComputePass.init();
    defer pass.deinit();
    const forms = [_]tile.Form{ .{ .nj = 1, .sgs = 8 }, .{ .nj = 1, .sgs = 16 }, .{ .nj = 2, .sgs = 4 }, .{ .nj = 2, .sgs = 8 } };
    const t = try std.heap.page_allocator.alloc(f64, n);
    defer std.heap.page_allocator.free(t);
    var differ: usize = 0;
    for (shapes) |sh| for ([_]u32{ 1, 2, 4, 5, 6, 7, 8, 12, 16 }) |rows| {
        std.debug.print("  fp8 {s:<9} {d:>5}x{d:<5} {d:>2} row(s): rows", .{ sh.name, sh.w.k, sh.w.n, rows });
        for (t) |*v| v.* = try once(r, junk, samples, pass, sh.w, x, ya, rows, .rows);
        std.debug.print(" {d:.1}", .{timing.least(t) * 1e6});
        if (rows >= 2) for (forms) |f| {
            if (f.sgs > (std.math.ceilPowerOfTwo(u32, sh.w.k / 32) catch unreachable)) continue;
            for (t) |*v| v.* = try once(r, junk, samples, pass, sh.w, x, yb, rows, .{ .tile = f });
            const mark: []const u8 = if (std.meta.eql(f, tile.formOf(sh.w.k, sh.w.n))) "*" else "";
            std.debug.print(" | c{d}s{d}{s} {d:.1}", .{ 8 * f.nj, f.sgs, mark, timing.least(t) * 1e6 });
            const bytes = 2 * @as(usize, rows) * sh.w.n;
            if (!std.mem.eql(u8, ya.contents()[0..bytes], yb.contents()[0..bytes])) differ += 1;
        };
        std.debug.print(" us\n", .{});
    };
    std.debug.print("{s} fp8 tile arms byte-equal to the row kernel: {d} differ\n", .{ if (differ == 0) "PASS" else "FAIL", differ });
    if (differ != 0) return error.TileDiffers;
}
