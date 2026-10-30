//! GPU smoke of tf_ling_route against route.zig on tiny and flash shapes.
const std = @import("std");
const mtl = @import("metal");
const route = @import("route");

const source = @import("ling_metal").route;
const Args = extern struct { rows: i32, ne: i32, topk: i32, ng: i32, keep: i32, scale: f32 };

fn check(s: route.Shape, rows: u32, logits: []const f32, bias: []const f32) !void {
    const n = rows * s.experts;
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
            const got_id = ids[r * s.topk + k];
            try std.testing.expect(got_id >= 0 and @as(u32, @intCast(got_id)) < s.experts);
            try std.testing.expectEqual(want[k].id, @as(u32, @intCast(got_id)));
            const gw = got[r * s.topk + k];
            const nan_ok = std.math.isNan(gw) and std.math.isNan(want[k].wt);
            try std.testing.expect(nan_ok or @abs(gw - want[k].wt) <= 1e-6);
            for (0..k) |j| try std.testing.expect(ids[r * s.topk + j] != got_id);
        }
    }
}

fn run(s: route.Shape, rows: u32, seed: u64, bias_shift: f32) !void {
    const a = std.testing.allocator;
    const n = rows * s.experts;
    const logits = try a.alloc(f32, n);
    defer a.free(logits);
    const bias = try a.alloc(f32, s.experts);
    defer a.free(bias);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (logits) |*v| v.* = rng.float(f32) * 4 - 2;
    for (bias) |*v| v.* = rng.float(f32) + bias_shift;
    try check(s, rows, logits, bias);
}

fn runTied(s: route.Shape, rows: u32) !void {
    const a = std.testing.allocator;
    const logits = try a.alloc(f32, rows * s.experts);
    defer a.free(logits);
    const bias = try a.alloc(f32, s.experts);
    defer a.free(bias);
    @memset(logits, 0);
    @memset(bias, 0);
    try check(s, rows, logits, bias);
}

test "tiny: 128 experts, 8 groups of 16, keep 4" {
    try run(.{ .experts = 128, .topk = 8, .n_group = 8, .topk_group = 4, .scale = 2.5 }, 8, 1, -0.5);
}

test "flash: 512 experts, 8 groups of 64, keep 4" {
    try run(.{ .experts = 512, .topk = 8, .n_group = 8, .topk_group = 4, .scale = 2.5 }, 4, 2, -0.5);
}

test "tiny: all choices negative, padded slots stay out of the picks" {
    try run(.{ .experts = 128, .topk = 8, .n_group = 8, .topk_group = 4, .scale = 2.5 }, 4, 3, -2.0);
}

const tiny = route.Shape{ .experts = 128, .topk = 8, .n_group = 8, .topk_group = 4, .scale = 2.5 };
const flash = route.Shape{ .experts = 512, .topk = 8, .n_group = 8, .topk_group = 4, .scale = 2.5 };

test "tiny: tied group scores keep the lower groups" {
    try runTied(tiny, 4);
}

test "flash: tied group scores keep the lower groups" {
    try runTied(flash, 2);
}

test "flash: planted NaN never wins a group or a pick" {
    const a = std.testing.allocator;
    const rows: u32 = 2;
    const logits = try a.alloc(f32, rows * flash.experts);
    defer a.free(logits);
    const bias = try a.alloc(f32, flash.experts);
    defer a.free(bias);
    @memset(logits, 0);
    @memset(bias, 0);
    logits[64] = std.math.nan(f32);
    try check(flash, rows, logits, bias);
}

fn oneRow(s: route.Shape, fill: f32, plants: []const struct { at: u32, v: f32 }) !void {
    const a = std.testing.allocator;
    const logits = try a.alloc(f32, s.experts);
    defer a.free(logits);
    const bias = try a.alloc(f32, s.experts);
    defer a.free(bias);
    @memset(logits, fill);
    @memset(bias, 0);
    for (plants) |p| logits[p.at] = p.v;
    try check(s, 1, logits, bias);
}

test "tiny: NaN last in the winning group keeps picks 0..7" {
    try oneRow(tiny, 0, &.{ .{ .at = 0, .v = 2 }, .{ .at = 1, .v = 2 }, .{ .at = 15, .v = std.math.nan(f32) } });
}

test "flash: NaN last in the winning group keeps picks 0..7" {
    try oneRow(flash, 0, &.{ .{ .at = 0, .v = 2 }, .{ .at = 1, .v = 2 }, .{ .at = 63, .v = std.math.nan(f32) } });
}

test "flash: odd-column NaN with a high even-column neighbour" {
    try oneRow(flash, 0, &.{ .{ .at = 17, .v = 3 }, .{ .at = 33, .v = std.math.nan(f32) } });
}

test "tiny and flash: all-NaN row, ids stay in range and distinct" {
    try oneRow(tiny, std.math.nan(f32), &.{});
    try oneRow(flash, std.math.nan(f32), &.{});
}

test "tiny: one finite logit per group" {
    try oneRow(tiny, std.math.nan(f32), &.{
        .{ .at = 3, .v = 1 },
        .{ .at = 19, .v = 1 },
        .{ .at = 35, .v = 1 },
        .{ .at = 51, .v = 1 },
        .{ .at = 67, .v = 1 },
        .{ .at = 83, .v = 1 },
        .{ .at = 99, .v = 1 },
        .{ .at = 115, .v = 1 },
    });
}

test "tiny: all -10 weights are 0.3125; all -100 weights are 0" {
    try oneRow(tiny, -10, &.{});
    try oneRow(tiny, -100, &.{});
}
