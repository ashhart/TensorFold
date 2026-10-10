//! Model-free real-GPU qualification; failures never fall back to a mock.
const std = @import("std");
const Runtime = @import("runtime.zig").Runtime;
const Context = @import("context.zig").Context;
const Stream = @import("stream.zig").Stream;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const HostBuffer = @import("memory.zig").HostBuffer;
const Module = @import("module.zig").Module;
const launch = @import("launch.zig");
const codeobject = @import("hip_probe");
const probe_image: @import("code_object.zig").Image = .{ .arch = codeobject.arch, .bytes = &codeobject.bytes };

test "HIP copies fills and mixed-width kernel arguments on real GPU" {
    var r = try Runtime.open();
    defer r.close();
    var ctx = try Context.init(&r, 0);
    defer ctx.deinit();
    try ctx.synchronize();
    var stream = try Stream.init(&r);
    defer stream.deinit();
    const n = 1025;
    var b = try DeviceBuffer.alloc(&r, n * @sizeOf(u32));
    defer b.free();
    var got: [n]u32 = undefined;
    const pattern: [n]u32 = @splat(0x12345678);
    try b.upload(0, std.mem.asBytes(&pattern));
    try b.download(0, std.mem.asBytes(&got));
    try std.testing.expectEqualSlices(u32, &pattern, &got);
    var pinned_src = try HostBuffer.alloc(&r, b.len);
    defer pinned_src.free();
    var pinned_dst = try HostBuffer.alloc(&r, b.len);
    defer pinned_dst.free();
    // Different from device contents: a no-op async upload must fail this check.
    @memset(pinned_src.bytes, 0x3c);
    @memset(pinned_dst.bytes, 0);
    try b.uploadAsync(0, pinned_src, stream);
    try b.downloadAsync(0, pinned_dst, stream);
    try stream.synchronize();
    try std.testing.expectEqualSlices(u8, pinned_src.bytes, pinned_dst.bytes);
    try b.fill8(0x5a);
    try b.download(0, std.mem.asBytes(&got));
    for (got) |v| try std.testing.expectEqual(@as(u32, 0x5a5a5a5a), v);
    try std.testing.expectError(error.Invalid, b.upload(b.len, &.{1}));
    var arch_buffer: [256]u8 = undefined;
    const arch = try @import("device_arch.zig").query(&r, 0, &arch_buffer);
    try std.testing.expect(@import("caps.zig").Caps.of(arch) != null);
    const images = [_]@import("code_object.zig").Image{.{ .arch = codeobject.arch, .bytes = &codeobject.bytes }};
    try std.testing.expectError(error.UnsupportedArchitecture, Module.loadForArchitecture(&r, &images, "gfx9999"));
    var m = try Module.loadForArchitecture(&r, &images, arch);
    defer m.unload();
    // No intervening host sync: the kernel must read the same-stream fill.
    const read_fill = try m.function("tf_hip_read_fill");
    var fill_args: launch.Args = .{};
    try fill_args.add(b.ptr);
    try fill_args.add(@as(u32, n));
    for ([_]u8{ 0xa5, 0x3c }) |value| {
        try b.fill8Async(value, stream);
        try launch.launch(read_fill, .{ .grid = .{ .x = 5 }, .block = .{ .x = 256 } }, stream, &fill_args);
        try stream.synchronize();
        try b.download(0, std.mem.asBytes(&got));
        const byte: u32 = value ^ 1;
        for (got) |v| try std.testing.expectEqual(byte * 0x01010101, v);
    }
    const f = try m.function("tf_hip_probe");
    var args: launch.Args = .{};
    try args.add(b.ptr);
    try args.add(@as(u32, n));
    try args.add(@as(u32, 17));
    try args.add(@as(u64, 0x0000000300000000));
    try launch.launch(f, .{ .grid = .{ .x = 5 }, .block = .{ .x = 256 } }, stream, &args);
    try stream.synchronize();
    try b.download(0, std.mem.asBytes(&got));
    for (got, 0..) |v, i| try std.testing.expectEqual(@as(u32, @intCast(i + 20)), v);
}

const graph = @import("graph.zig");

fn fillConfig(n: u32) launch.Config {
    return .{ .grid = .{ .x = (n + 255) / 256 }, .block = .{ .x = 256 } };
}

test "device copies, word fills, mapped host memory, a module global and wave32" {
    var r = try Runtime.open();
    defer r.close();
    var ctx = try Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try Stream.init(&r);
    defer stream.deinit();
    var arch_buffer: [256]u8 = undefined;
    var m = try Module.loadForArchitecture(&r, &.{probe_image}, try @import("device_arch.zig").query(&r, 0, &arch_buffer));
    defer m.unload();
    const held = @import("memory.zig").usage(true).device;
    var pattern: [1024]u32 = undefined;
    for (&pattern, 0..) |*p, i| p.* = @intCast(i);
    var a = try DeviceBuffer.fromHost(&r, std.mem.asBytes(&pattern));
    defer a.free();
    var b = try DeviceBuffer.alloc(&r, a.len);
    defer b.free();
    try std.testing.expectEqual(held + 2 * a.len, @import("memory.zig").usage(false).device);
    var got: [1024]u32 = undefined;
    try b.copyFrom(0, try a.address(0), a.len);
    try b.download(0, std.mem.asBytes(&got));
    try std.testing.expectEqualSlices(u32, &pattern, &got);
    try b.fill32(0xdeadbeef);
    try b.copyFromAsync(4, try a.address(8), 8, stream);
    try stream.synchronize();
    try b.download(0, std.mem.asBytes(&got));
    try std.testing.expectEqualSlices(u32, &.{ 0xdeadbeef, 2, 3, 0xdeadbeef }, got[0..4]);
    try b.fill32Async(7, stream);
    try stream.synchronize();
    try b.download(0, std.mem.asBytes(&got));
    for (got) |v| try std.testing.expectEqual(@as(u32, 7), v);

    // A kernel writes pinned host memory directly; no copy reads it back.
    var mapped = try HostBuffer.allocMapped(&r, 1024 * @sizeOf(f32));
    defer mapped.free();
    @memset(mapped.bytes, 0);
    var args: launch.Args = .{};
    try args.add(try mapped.device());
    try args.add(@as(f32, 7));
    try args.add(@as(u32, 1024));
    try launch.launch(try m.function("tf_hip_fill_f32"), fillConfig(1024), stream, &args);
    try stream.synchronize();
    for (std.mem.bytesAsSlice(f32, mapped.bytes), 0..) |v, i| try std.testing.expectEqual(7 + @as(f32, @floatFromInt(i)), v);

    const table = try m.global("tf_hip_table");
    try std.testing.expectEqual(@as(usize, 16), table.len);
    const values = [4]i32{ 1, 2, 3, 4 };
    try runtime_check(r.api.hipMemcpyHtoD(@ptrFromInt(table.address), std.mem.asBytes(&values), 16));
    args = .{};
    try args.add(b.ptr);
    try launch.launch(try m.function("tf_hip_read_table"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 4 } }, stream, &args);
    try stream.synchronize();
    var doubled: [4]i32 = undefined;
    try b.download(0, std.mem.asBytes(&doubled));
    try std.testing.expectEqualSlices(i32, &.{ 2, 4, 6, 8 }, &doubled);
    try std.testing.expectError(error.HipFailed, m.global("tf_hip_missing"));

    try launch.launch(try m.function("tf_hip_wave"), .{ .grid = .{ .x = 1 }, .block = .{ .x = 64 } }, stream, &args);
    try stream.synchronize();
    var wave: i32 = 0;
    try b.download(0, std.mem.asBytes(&wave));
    try std.testing.expectEqual(@as(i32, 32), wave);
}

const runtime_check = @import("runtime.zig").check;

test "a captured stream replays, and its nodes take new arguments in place" {
    var r = try Runtime.open();
    defer r.close();
    var ctx = try Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try Stream.init(&r);
    defer stream.deinit();
    var arch_buffer: [256]u8 = undefined;
    var m = try Module.loadForArchitecture(&r, &.{probe_image}, try @import("device_arch.zig").query(&r, 0, &arch_buffer));
    defer m.unload();
    const step = try m.function("tf_hip_step");
    var counter = try DeviceBuffer.alloc(&r, 8);
    defer counter.free();
    try counter.fill8(0);
    const one: launch.Config = .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } };
    try graph.beginCapture(stream, .thread_local);
    try std.testing.expectEqual(@import("abi.zig").CaptureStatus.active, try graph.captureStatus(stream));
    for (1..9) |i| {
        var args: launch.Args = .{};
        try args.add(counter.ptr);
        try args.add(@as(u64, i));
        try launch.launch(step, one, stream, &args);
    }
    var captured = try graph.endCapture(stream);
    defer captured.deinit();
    var buf: [16]graph.Node = undefined;
    const nodes = try captured.nodes(&buf);
    try std.testing.expectEqual(@as(usize, 8), nodes.len);
    var total: u64 = 0;
    try counter.download(0, std.mem.asBytes(&total));
    try std.testing.expectEqual(@as(u64, 0), total);
    var exec = try captured.instantiate();
    defer exec.deinit();
    try exec.upload(stream);
    for (0..3) |_| try exec.launchOn(stream);
    try stream.synchronize();
    try counter.download(0, std.mem.asBytes(&total));
    try std.testing.expectEqual(@as(u64, 3 * 36), total);
    for (nodes) |node| {
        var args: launch.Args = .{};
        try args.add(counter.ptr);
        try args.add(@as(u64, 10));
        try exec.setKernel(node, step, one, &args);
    }
    try exec.launchOn(stream);
    try stream.synchronize();
    try counter.download(0, std.mem.asBytes(&total));
    try std.testing.expectEqual(@as(u64, 3 * 36 + 80), total);
}

fn fillAxpy(g: graph.Graph, fill: @import("module.zig").Function, axpy: @import("module.zig").Function, y: DeviceBuffer, x: DeviceBuffer, base: f32, a: f32, n: u32) !void {
    var fa: launch.Args = .{};
    try fa.add(y.ptr);
    try fa.add(base);
    try fa.add(n);
    const first = try g.addKernel(&.{}, fill, fillConfig(n), &fa);
    var aa: launch.Args = .{};
    try aa.add(y.ptr);
    try aa.add(x.ptr);
    try aa.add(a);
    try aa.add(n);
    _ = try g.addKernel(&.{first}, axpy, fillConfig(n), &aa);
}

test "an explicit graph keeps its edge, updates from a same-topology graph and refuses a topology change" {
    var r = try Runtime.open();
    defer r.close();
    var ctx = try Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try Stream.init(&r);
    defer stream.deinit();
    var arch_buffer: [256]u8 = undefined;
    var m = try Module.loadForArchitecture(&r, &.{probe_image}, try @import("device_arch.zig").query(&r, 0, &arch_buffer));
    defer m.unload();
    const fill = try m.function("tf_hip_fill_f32");
    const axpy = try m.function("tf_hip_axpy");
    const n: u32 = 4096;
    var x = try DeviceBuffer.alloc(&r, n * 4);
    defer x.free();
    var y = try DeviceBuffer.alloc(&r, n * 4);
    defer y.free();
    var xa: launch.Args = .{};
    try xa.add(x.ptr);
    try xa.add(@as(f32, 0));
    try xa.add(n);
    try launch.launch(fill, fillConfig(n), stream, &xa);
    try stream.synchronize();
    var g1 = try graph.Graph.init(&r);
    defer g1.deinit();
    try fillAxpy(g1, fill, axpy, y, x, 1, 2, n);
    var exec = try g1.instantiate();
    defer exec.deinit();
    try exec.launchOn(stream);
    try stream.synchronize();
    var got: [n]f32 = undefined;
    try y.download(0, std.mem.sliceAsBytes(&got));
    // y = 2x + (1 + i) with x = i
    for (got, 0..) |v, i| try std.testing.expectEqual(3 * @as(f32, @floatFromInt(i)) + 1, v);
    var g2 = try graph.Graph.init(&r);
    defer g2.deinit();
    try fillAxpy(g2, fill, axpy, y, x, 5, 3, n);
    try std.testing.expectEqual(@import("abi.zig").ExecUpdateResult.success, try exec.update(g2));
    try exec.launchOn(stream);
    try stream.synchronize();
    try y.download(0, std.mem.sliceAsBytes(&got));
    for (got, 0..) |v, i| try std.testing.expectEqual(4 * @as(f32, @floatFromInt(i)) + 5, v);
    var g3 = try graph.Graph.init(&r);
    defer g3.deinit();
    try fillAxpy(g3, fill, axpy, y, x, 7, 4, n);
    _ = try g3.addKernel(&.{}, fill, fillConfig(n), &xa);
    try std.testing.expect(try exec.update(g3) != .success);
    try exec.launchOn(stream);
    try stream.synchronize();
    try y.download(0, std.mem.sliceAsBytes(&got));
    for (got, 0..) |v, i| try std.testing.expectEqual(4 * @as(f32, @floatFromInt(i)) + 5, v);
}

test "a cooperative grid of one block a compute unit, and the device's attributes" {
    var r = try Runtime.open();
    defer r.close();
    var ctx = try Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try Stream.init(&r);
    defer stream.deinit();
    var name_buf: [256]u8 = undefined;
    try std.testing.expect((try ctx.name(&name_buf)).len > 0);
    try std.testing.expectEqual(@as(c_int, 32), try ctx.attribute(.warp_size));
    try std.testing.expect(try ctx.attribute(.cooperative_launch) != 0);
    const cus: u32 = @intCast(try ctx.attribute(.multiprocessor_count));
    var arch_buffer: [256]u8 = undefined;
    var m = try Module.loadForArchitecture(&r, &.{probe_image}, try @import("device_arch.zig").query(&r, 0, &arch_buffer));
    defer m.unload();
    const fill = try m.function("tf_hip_fill_f32");
    try std.testing.expect(try fill.occupancy(256, 0) >= 1);
    try std.testing.expect(try fill.attribute(.max_threads_per_block) >= 256);
    var y = try DeviceBuffer.alloc(&r, cus * 256 * 4);
    defer y.free();
    var args: launch.Args = .{};
    try args.add(y.ptr);
    try args.add(@as(f32, 3));
    try args.add(cus * 256);
    try launch.launch(fill, .{ .grid = .{ .x = cus }, .block = .{ .x = 256 }, .cooperative = true }, stream, &args);
    try stream.synchronize();
    const got = try std.testing.allocator.alloc(f32, cus * 256);
    defer std.testing.allocator.free(got);
    try y.download(0, std.mem.sliceAsBytes(got));
    for (got, 0..) |v, i| try std.testing.expectEqual(3 + @as(f32, @floatFromInt(i)), v);
    try std.testing.expect(r.errorName(1).len > 0);
}
