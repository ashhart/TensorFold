//! Vector add, zero-copy buffers over mmap, and residency sets.
const std = @import("std");
const mtl = @import("metal");
const common = @import("common.zig");

const Gpu = common.Gpu;
const Size = mtl.Size;

/// c = a + b on `n` floats through tf_vadd.
fn vadd(gpu: Gpu, a: mtl.Buffer, b: mtl.Buffer, c: mtl.Buffer, n: u32) !void {
    const p = try gpu.pipeline("tf_vadd", false);
    defer p.deinit();
    const cb = gpu.queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(p);
    enc.setBuffer(a, 0, 0);
    enc.setBuffer(b, 0, 1);
    enc.setBuffer(c, 0, 2);
    enc.setValue(n, 3);
    enc.dispatchThreads(Size.of(n, 1, 1), Size.of(256, 1, 1));
    enc.end();
    try common.run(cb);
}

pub fn vectorAdd(gpu: Gpu) !void {
    const n: u32 = (1 << 20) + 37;
    const a = try gpu.buffer(f32, n);
    defer a.deinit();
    const b = try gpu.buffer(f32, n);
    defer b.deinit();
    const c = try gpu.buffer(f32, n);
    defer c.deinit();
    const av = a.slice(f32, n);
    const bv = b.slice(f32, n);
    for (av, bv, 0..) |*x, *y, i| {
        x.* = @as(f32, @floatFromInt(i)) * 0.5;
        y.* = 3.0 - @as(f32, @floatFromInt(i));
    }
    try vadd(gpu, a, b, c, n);
    for (c.slice(f32, n), av, bv) |z, x, y| if (z != x + y) return error.Mismatch;
}

/// A GPU buffer over host pages with no copy: the GPU reads the pages the host writes.
pub fn zeroCopy(gpu: Gpu) !void {
    const page = std.heap.pageSize();
    const len = 4 * page;
    const n: u32 = @intCast(len / @sizeOf(f32));
    const mem = try std.posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(mem);
    const host: []f32 = @as([*]f32, @ptrCast(mem.ptr))[0..n];
    const a = try gpu.device.bufferNoCopy(mem.ptr, len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer a.deinit();
    if (a.contents() != mem.ptr) return error.Copied;
    const c = try gpu.buffer(f32, n);
    defer c.deinit();
    for (0..2) |pass| {
        for (host, 0..) |*x, i| x.* = @floatFromInt(i * (pass + 1));
        try vadd(gpu, a, a, c, n);
        for (c.slice(f32, n), host) |z, x| if (z != x + x) return error.Mismatch;
    }
}

pub fn residencySet(gpu: Gpu) !void {
    const set = gpu.device.residencySet(4) catch |err| {
        if (err == error.Unsupported) {
            std.debug.print("  residency sets need macOS 15: skipped\n", .{});
            return;
        }
        return err;
    };
    defer set.deinit();
    const n: u32 = 1 << 16;
    const a = try gpu.buffer(f32, n);
    defer a.deinit();
    const c = try gpu.buffer(f32, n);
    defer c.deinit();
    for (a.slice(f32, n), 0..) |*x, i| x.* = @floatFromInt(i);
    set.add(a);
    set.add(c);
    set.commit();
    set.requestResidency();
    gpu.queue.addResidencySet(set);
    defer gpu.queue.removeResidencySet(set);
    try vadd(gpu, a, a, c, n);
    for (c.slice(f32, n), a.slice(f32, n)) |z, x| if (z != x + x) return error.Mismatch;
    std.debug.print("  {d} allocations, {d} KB resident\n", .{ set.count(), set.allocatedSize() / 1024 });
    if (set.count() != 2) return error.WrongCount;
}
