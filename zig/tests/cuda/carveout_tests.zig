//! The display carveout on a GB10: host and kernel round trips through the registered framebuffer, and its bandwidth.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;
const expect = check.expect;

/// `carveout [MiB]`: SKIP when the card cannot be opened; FAIL when it opens but the driver refuses a step.
pub fn run(gpu: Gpu, card: [:0]const u8, mib: usize) !void {
    const before = try memAvailable(gpu.io);
    var c = cuda.Carveout.open(gpu.d, card, mib << 20) catch |err| switch (err) {
        error.CardUnavailable => {
            std.debug.print("SKIP carveout: {s} cannot be opened (no DRM card, or no access to it)\n", .{card});
            return;
        },
        else => return err,
    };
    defer c.close();
    const spent = before -| try memAvailable(gpu.io);
    const flags = if (c.flags & cuda.abi.host_register_iomemory != 0) "DEVICEMAP|IOMEMORY" else "DEVICEMAP";
    check.pass("carveout: {d} MiB from {s} registered {s} at 0x{x}, MemAvailable moved {d} MiB", .{ c.bytes() >> 20, card, flags, c.dev, spent >> 20 });
    try roundTrips(gpu, &c);
    try bandwidth(gpu, &c);
    try expect(c.freeBytes() == c.bytes(), "every span given back: {d} of {d} bytes free", .{ c.freeBytes(), c.bytes() });
}

/// Host writes the mapping, a device copy reads it; a kernel writes a span, the host reads the mapping.
fn roundTrips(gpu: Gpu, c: *cuda.Carveout) !void {
    const n: usize = 1 << 20;
    const span = (try c.take(n)) orelse return error.TestFailed;
    defer c.give(span);
    const at = span.ptr - c.dev;
    try expect(at % cuda.carveout.alignment == 0, "span offset {d} is 256-byte aligned", .{at});
    const host = c.host[at..][0..n];
    try expect(std.mem.allEqual(u8, host, 0), "a new span reads zero from the host", .{});
    for (host, 0..) |*p, i| p.* = @truncate(i *% 2654435761 >> 7);
    var dev = try cuda.DeviceBuffer.alloc(gpu.d, n);
    defer dev.free();
    try dev.copyFrom(0, span.ptr, n, null);
    const back = try check.download(gpu, dev);
    defer gpu.gpa.free(back);
    try check.sameBytes("host write, device read (1 MiB)", back, host);
    check.pass("host write -> device copy -> host: 1 MiB equal", .{});

    var probe = try cuda.Module.load(gpu.d, cuda.kernels.probe);
    defer probe.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const count: u32 = @intCast(n / 4);
    var args: cuda.Args = .{};
    args.add(span.ptr);
    args.add(@as(f32, 7));
    args.add(count);
    try cuda.launch.launch(try probe.function("tf_probe_fill"), .{ .grid = .{ .x = (count + 255) / 256 }, .block = .{ .x = 256 } }, stream, &args);
    try stream.synchronize();
    const words: []align(1) const f32 = std.mem.bytesAsSlice(f32, host);
    for (words, 0..) |v, i| try expect(v == 7 + @as(f32, @floatFromInt(i)), "kernel write, host read: element {d} is {d}", .{ i, v });
    check.pass("kernel write -> host read through the mapping: {d} floats equal", .{count});
}

/// Copy GB/s into and out of the carveout against device memory, and a kernel reading its input from either.
fn bandwidth(gpu: Gpu, c: *cuda.Carveout) !void {
    const n: usize = @min(c.freeBytes(), 1 << 30) / (1 << 20) * (1 << 20);
    const span = (try c.take(n)) orelse return error.TestFailed;
    defer c.give(span);
    var a = try cuda.DeviceBuffer.alloc(gpu.d, n);
    defer a.free();
    var b = try cuda.DeviceBuffer.alloc(gpu.d, n);
    defer b.free();
    try a.fill8(0x5a, null);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();
    const Pair = struct { dst: u64, src: u64 };
    const cases = [_]Pair{ .{ .dst = span.ptr, .src = a.ptr }, .{ .dst = b.ptr, .src = span.ptr }, .{ .dst = b.ptr, .src = a.ptr } };
    var rates: [3]f64 = undefined;
    for (cases, &rates) |p, *r| {
        const reps = 5;
        try gpu.d.check(gpu.d.api.cuMemcpyDtoDAsync_v2(p.dst, p.src, n, stream.handle), "warm copy");
        try t0.record(stream);
        for (0..reps) |_| try gpu.d.check(gpu.d.api.cuMemcpyDtoDAsync_v2(p.dst, p.src, n, stream.handle), "copy");
        try t1.record(stream);
        try t1.synchronize();
        r.* = reps * @as(f64, @floatFromInt(n)) / (try cuda.Event.elapsedMs(t0, t1) * 1e6);
    }
    var probe = try cuda.Module.load(gpu.d, cuda.kernels.probe);
    defer probe.unload();
    const axpy = try probe.function("tf_probe_axpy");
    const count: u32 = @intCast(n / 4);
    var reads: [2]f64 = undefined;
    for ([_]u64{ span.ptr, a.ptr }, &reads) |x, *r| {
        const reps = 5;
        for (0..reps + 1) |i| {
            if (i == 1) try t0.record(stream);
            var args: cuda.Args = .{};
            args.add(b.ptr);
            args.add(x);
            args.add(@as(f32, 0.5));
            args.add(count);
            try cuda.launch.launch(axpy, .{ .grid = .{ .x = (count + 255) / 256 }, .block = .{ .x = 256 } }, stream, &args);
        }
        try t1.record(stream);
        try t1.synchronize();
        r.* = reps * 3 * @as(f64, @floatFromInt(n)) / (try cuda.Event.elapsedMs(t0, t1) * 1e6);
    }
    std.debug.print("RESULT carveout {d} MiB copies GB/s: into {d:.0}, out of {d:.0}, device->device {d:.0}; axpy GB/s reading x from the carveout {d:.0}, from device memory {d:.0}\n", .{ n >> 20, rates[0], rates[1], rates[2], reads[0], reads[1] });
    check.pass("bandwidth probe over {d} MiB", .{n >> 20});
}

/// /proc/meminfo's MemAvailable in bytes.
fn memAvailable(io: std.Io) !usize {
    var buf: [8192]u8 = undefined;
    const text = try std.Io.Dir.cwd().readFile(io, "/proc/meminfo", &buf);
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "MemAvailable:")) continue;
        var words = std.mem.tokenizeScalar(u8, line["MemAvailable:".len..], ' ');
        return 1024 * try std.fmt.parseInt(usize, words.next() orelse return error.BadMeminfo, 10);
    }
    return error.BadMeminfo;
}
