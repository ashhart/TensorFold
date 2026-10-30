//! GPU smoke of tf_ling_route against route.zig on tiny and flash shapes.
const std = @import("std");
const mtl = @import("metal");
const route = @import("route");

const source = @import("ling_metal").route;
const Args = extern struct { rows: i32, ne: i32, topk: i32, ng: i32, keep: i32, scale: f32 };

fn run(s: route.Shape, rows: u32, seed: u64) !void {
    const a = std.testing.allocator;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, "tf_ling_route", false);
    defer pipe.deinit();
    const queue = try device.queue();
    defer queue.deinit();

    const n = rows * s.experts;
    var logits = try a.alloc(f32, n);
    defer a.free(logits);
    const bias = try a.alloc(f32, s.experts);
    defer a.free(bias);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (logits) |*v| v.* = rng.float(f32) * 4 - 2;
    for (bias) |*v| v.* = rng.float(f32) - 0.5;

    const x = try device.buffer(n * 4, mtl.ResourceOptions.shared);
    defer x.deinit();
    const b = try device.buffer(s.experts * 4, mtl.ResourceOptions.shared);
    defer b.deinit();
    const pick = try device.buffer(rows * s.topk * 4, mtl.ResourceOptions.shared);
    defer pick.deinit();
    const wts = try device.buffer(rows * s.topk * 4, mtl.ResourceOptions.shared);
    defer wts.deinit();
    @memcpy(std.mem.bytesAsSlice(f32, x.contents()[0 .. n * 4]), logits);
    @memcpy(std.mem.bytesAsSlice(f32, b.contents()[0 .. s.experts * 4]), bias);

    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(pipe);
    e.setBuffer(x, 0, 0);
    e.setBuffer(b, 0, 1);
    e.setValue(Args{
        .rows = @intCast(rows),
        .ne = @intCast(s.experts),
        .topk = @intCast(s.topk),
        .ng = @intCast(s.n_group),
        .keep = @intCast(s.topk_group),
        .scale = s.scale,
    }, 2);
    e.setBuffer(pick, 0, 3);
    e.setBuffer(wts, 0, 4);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("GPU failed: {s}\n", .{why});
        return error.GpuFailed;
    }

    const ids = std.mem.bytesAsSlice(i32, pick.contents()[0 .. rows * s.topk * 4]);
    const got = std.mem.bytesAsSlice(f32, wts.contents()[0 .. rows * s.topk * 4]);
    var want: [route.max_topk]route.Pick = undefined;
    for (0..rows) |r| {
        route.selectF32(logits[r * s.experts ..][0..s.experts], bias, want[0..s.topk], s);
        for (0..s.topk) |k| {
            try std.testing.expectEqual(want[k].id, @as(u32, @intCast(ids[r * s.topk + k])));
            try std.testing.expect(@abs(got[r * s.topk + k] - want[k].wt) < 1e-5);
        }
    }
}

test "tiny: 128 experts, 8 groups of 16, keep 4" {
    try run(.{ .experts = 128, .topk = 8, .n_group = 8, .topk_group = 4, .scale = 2.5 }, 8, 1);
}

test "flash: 512 experts, 8 groups of 64, keep 4" {
    try run(.{ .experts = 512, .topk = 8, .n_group = 8, .topk_group = 4, .scale = 2.5 }, 4, 2);
}
