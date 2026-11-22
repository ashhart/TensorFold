//! TF_Q27_CONV_INPLACE's contract cells: the in-place kernel against conv_commit then publish_conv, every history byte.
const std = @import("std");
const mtl = @import("metal");
const delta = @import("core_delta");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const Params = extern struct { rows: u32, nk: u32, nv: u32, dk: u32, dv: u32, taps: u32, slots: u32, weight_flags: u32, eps: f32, zba_stride: u32 = 0 };
const Keep = extern struct { first: u32, rows: u32, state_slot: u32, next_slot: u32 };
const nk = 1;
const nv = 2;
const dk = 128;
const dv = 128;
const channels = 2 * nk * dk + nv * dv;
const slots = 4;
const max_rows = 16;

/// A tree of rows and the root-to-leaf path each of up to two slots keeps (an empty path keeps its window).
const Case = struct { name: []const u8, taps: u32, parents: []const i32, paths: [2][]const u32 };
const cases = [_]Case{
    .{ .name = "one kept row, old taps shift down", .taps = 4, .parents = &.{-1}, .paths = .{ &.{0}, &.{} } },
    .{ .name = "two kept rows", .taps = 4, .parents = &.{ -1, 0 }, .paths = .{ &.{ 0, 1 }, &.{} } },
    .{ .name = "chain as long as the window", .taps = 4, .parents = &.{ -1, 0, 1 }, .paths = .{ &.{ 0, 1, 2 }, &.{} } },
    .{ .name = "chain past the window", .taps = 4, .parents = &.{ -1, 0, 1, 2, 3, 4, 5, 6 }, .paths = .{ &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, &.{} } },
    .{ .name = "branch kept", .taps = 4, .parents = &.{ -1, 0, 0, 2, 2 }, .paths = .{ &.{ 0, 2, 4 }, &.{} } },
    .{ .name = "empty keep", .taps = 4, .parents = &.{-1}, .paths = .{ &.{}, &.{} } },
    .{ .name = "two slots, one empty", .taps = 4, .parents = &.{ -1, 0 }, .paths = .{ &.{ 0, 1 }, &.{} } },
    .{ .name = "two taps", .taps = 2, .parents = &.{ -1, 0, 1 }, .paths = .{ &.{ 0, 1, 2 }, &.{0} } },
    .{ .name = "eight taps", .taps = 8, .parents = &.{ -1, 0, 1 }, .paths = .{ &.{ 0, 1, 2 }, &.{0} } },
};

fn run(queue: mtl.Queue, encode: anytype) !void {
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    encode.go(e);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |message| {
        std.debug.print("GPU failed: {s}\n", .{message});
        return error.GpuFailed;
    }
}

const Buffers = struct { conv: mtl.Buffer, next: mtl.Buffer, projection: mtl.Buffer, kept: mtl.Buffer, keeps: mtl.Buffer, windows: mtl.Buffer };
const Twice = struct {
    k: delta.Kernels,
    b: Buffers,
    p: Params,
    width: usize,
    fn go(x: @This(), e: mtl.ComputeEncoder) void {
        e.setPipeline(x.k.pipelines.conv_commit);
        for ([_]mtl.Buffer{ x.b.conv, x.b.projection, x.b.kept, x.b.keeps, x.b.windows }, 0..) |b, i| e.setBuffer(b, 0, i);
        e.setValue(x.p, 5);
        e.setBuffer(x.b.next, 0, 6);
        e.dispatchThreads(.{ .width = x.width, .height = 2 }, .{ .width = 32 });
        e.barrier();
        e.setPipeline(x.k.publish_conv);
        e.setBuffer(x.b.next, 0, 0);
        e.setBuffer(x.b.keeps, 0, 1);
        e.setValue(x.p, 2);
        e.setBuffer(x.b.conv, 0, 3);
        e.dispatchThreads(.{ .width = x.width, .height = 2 }, .{ .width = 32 });
    }
};
const InPlace = struct {
    k: delta.Kernels,
    b: Buffers,
    p: Params,
    fn go(x: @This(), e: mtl.ComputeEncoder) void {
        e.setPipeline(x.k.conv_in_place);
        for ([_]mtl.Buffer{ x.b.conv, x.b.projection, x.b.kept, x.b.keeps, x.b.windows }, 0..) |b, i| e.setBuffer(b, 0, i);
        e.setValue(x.p, 5);
        e.dispatchThreads(.{ .width = channels, .height = 2 }, .{ .width = 32 });
    }
};

fn check(queue: mtl.Queue, k: delta.Kernels, b: Buffers, copy: mtl.Buffer, c: Case, seed: u64) !bool {
    const kept: usize = c.taps - 1;
    const width = kept * channels;
    const words = slots * width;
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    for (b.conv.slice(u16, words)) |*x| x.* = r.int(u16); // every bit pattern, NaNs included
    for (b.projection.slice(u16, max_rows * channels)) |*x| x.* = r.int(u16);
    @memcpy(copy.slice(u16, words), b.conv.slice(u16, words));
    const windows = b.windows.slice(u32, max_rows * 8);
    for (c.parents, 0..) |_, row| for (0..c.taps) |tap| {
        const behind = c.taps - 1 - tap;
        var ancestor: i32 = @intCast(row);
        var hops: usize = 0;
        while (hops < behind and ancestor >= 0) : (hops += 1) ancestor = c.parents[@intCast(ancestor)];
        windows[row * c.taps + tap] = @intCast(if (ancestor >= 0) kept + @as(usize, @intCast(ancestor)) else kept - (behind - hops + 1));
    };
    const list = b.kept.slice(u32, max_rows);
    @memcpy(list[0..c.paths[0].len], c.paths[0]);
    @memcpy(list[c.paths[0].len..][0..c.paths[1].len], c.paths[1]);
    b.keeps.slice(Keep, 2)[0] = .{ .first = 0, .rows = @intCast(c.paths[0].len), .state_slot = 0, .next_slot = 0 };
    b.keeps.slice(Keep, 2)[1] = .{ .first = @intCast(c.paths[0].len), .rows = @intCast(c.paths[1].len), .state_slot = 2, .next_slot = 2 };
    const p = Params{ .rows = @intCast(c.parents.len), .nk = nk, .nv = nv, .dk = dk, .dv = dv, .taps = c.taps, .slots = slots, .weight_flags = 0, .eps = 1e-6 };
    try run(queue, Twice{ .k = k, .b = b, .p = p, .width = width });
    var other = b;
    other.conv = copy;
    try run(queue, InPlace{ .k = k, .b = other, .p = p });
    return std.mem.eql(u16, b.conv.slice(u16, words), copy.slice(u16, words));
}

pub fn main() !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const k = try delta.Kernels.init(device);
    defer k.deinit();
    const history = slots * 7 * channels * 2;
    const b = Buffers{ .conv = try device.buffer(history, opts), .next = try device.buffer(history, opts), .projection = try device.buffer(max_rows * channels * 2, opts), .kept = try device.buffer(max_rows * 4, opts), .keeps = try device.buffer(2 * @sizeOf(Keep), opts), .windows = try device.buffer(max_rows * 8 * 4, opts) };
    const copy = try device.buffer(history, opts);
    std.debug.print("device: {s}\n", .{device.name()});
    var failed: usize = 0;
    for (cases, 0..) |c, i| {
        var bad: usize = 0;
        for (0..16) |round| {
            if (!try check(queue, k, b, copy, c, 0x2715_0000 + i * 256 + round)) bad += 1;
        }
        std.debug.print("{s}: {s}\n", .{ c.name, if (bad == 0) "every conv byte equal" else "FAILED" });
        failed += bad;
    }
    if (failed > 0) {
        std.debug.print("{d} mismatches\n", .{failed});
        std.process.exit(1);
    }
    std.debug.print("all conv cells match\n", .{});
}
