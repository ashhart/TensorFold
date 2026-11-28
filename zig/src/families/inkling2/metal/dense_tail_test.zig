//! The dense simdgroup-matrix kernel's last row tile never reads past the input's last row.
const std = @import("std");
const mtl = @import("metal");
const src = @import("inkling2_metal");
const dense = @import("dense.zig");

const k = 512;
const n = 16;
const extra_rows = 16;
const canary: u16 = 0x7fc0;
const poison: u16 = 0xbeef;

fn isNan(bits: u16) bool {
    return bits & 0x7fff > 0x7f80;
}

/// The kernel with every row of the tiles it computes stored, so a row's value shows what it read.
fn witness(a: std.mem.Allocator, guarded: bool) ![]u8 {
    const store = "if (row < a.rows) Y[";
    if (std.mem.count(u8, src.dense_mma, store) != 1) return error.WitnessEditCount;
    const wide = try std.mem.replaceOwned(u8, a, src.dense_mma, store, "if (row < a.rows + 16u) Y[");
    if (guarded) return wide;
    defer a.free(wide);
    const tail = "if (r0 + 8u * i + 8u <= a.rows) {";
    if (std.mem.count(u8, wide, tail) != 1) return error.WitnessEditCount;
    return std.mem.replaceOwned(u8, a, wide, tail, "if (true) {");
}

fn run(device: mtl.Device, queue: mtl.Queue, text: []const u8, rows: u32, y: []u16) !void {
    const mma = try dense.Mma.loadSource(device, text);
    defer mma.deinit();
    const x = try device.buffer(2 * k * (rows + extra_rows), mtl.ResourceOptions.shared);
    defer x.deinit();
    const w = try device.buffer(2 * n * k, mtl.ResourceOptions.shared);
    defer w.deinit();
    const out = try device.buffer(2 * n * (rows + extra_rows), mtl.ResourceOptions.shared);
    defer out.deinit();
    var prng = std.Random.DefaultPrng.init(rows);
    const random = prng.random();
    const xs = x.slice(u16, k * (rows + extra_rows));
    for (xs[0 .. k * rows]) |*v| v.* = (@as(u16, random.int(u8) & 0x3f) | 0x3f00) ^ (@as(u16, random.int(u1)) << 15);
    @memset(xs[k * rows ..], canary);
    for (w.slice(u16, n * k)) |*v| v.* = (@as(u16, random.int(u8) & 0x3f) | 0x3f00) ^ (@as(u16, random.int(u1)) << 15);
    @memset(out.slice(u16, n * (rows + extra_rows)), poison);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    mma.launch(e, .{ .buf = x }, .{ .buf = w }, .{ .buf = out }, .{ .rows = rows, .n = n, .k = k, .ldy = n });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
    @memcpy(y, out.slice(u16, n * (rows + extra_rows)));
}

test "the dense kernel's tail tile reads the last row, not the memory after the input" {
    const a = std.testing.allocator;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const guarded = try witness(a, true);
    defer a.free(guarded);
    const unguarded = try witness(a, false);
    defer a.free(unguarded);
    for ([_]u32{ 1, 7, 9, 45, 255 }) |rows| {
        const want = try a.alloc(u16, n * (rows + extra_rows));
        defer a.free(want);
        const old = try a.alloc(u16, want.len);
        defer a.free(old);
        try run(device, queue, guarded, rows, want);
        try run(device, queue, unguarded, rows, old);
        const last = (rows - 1) / 16 * 16;
        const covered = last + (if (rows - last > 8) @as(u32, 16) else 8);
        try std.testing.expect(covered > rows);
        // The rows a tile computes are all finite, the unstored ones repeat the last row, and nothing else is written.
        var old_nan = false;
        for (0..rows + extra_rows) |row| {
            const got = want[row * n ..][0..n];
            if (row < rows) {
                try std.testing.expectEqualSlices(u16, old[row * n ..][0..n], got);
            } else if (row < covered) {
                try std.testing.expectEqualSlices(u16, want[(rows - 1) * n ..][0..n], got);
                for (old[row * n ..][0..n]) |v| old_nan = old_nan or isNan(v);
            } else for (got) |v| try std.testing.expectEqual(poison, v);
        }
        // The planted control: the unguarded loads do read the canary, so the check above can fail.
        try std.testing.expect(old_nan);
    }
}
