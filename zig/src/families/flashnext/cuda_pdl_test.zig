//! Record the production CUDA boundary: changing PDL changes only the launch attribute.
const std = @import("std");
const t = std.testing;
const cuda = @import("cuda");
const policy = @import("cuda_pdl.zig");
const Glue = @import("cuda_glue.zig").Glue;
const Kernels = @import("cuda_batch_kernels.zig").Kernels;
const ops = @import("cuda_batch_ops.zig");
const gdn = @import("cuda_gdn_fused.zig");
const env = @import("sys").env;

const Record = struct {
    grid: [3]u32,
    block: [3]u32,
    shared: u32,
    stream: cuda.abi.Stream,
    function: cuda.abi.Function,
    pdl: bool,
    bytes: [256]u8 = @splat(0),
    used: usize = 0,
};
const Mock = struct {
    var got: Record = undefined;
    var sizes: []const u16 = &.{};
    var calls: usize = 0;
    var capability_calls: usize = 0;
    var major: c_int = 12;
    var minor: c_int = 1;
    fn record(f: cuda.abi.Function, grid: [3]u32, block: [3]u32, shared: u32, s: cuda.abi.Stream, args: ?[*]?*anyopaque, pdl: bool) cuda.abi.Result {
        got = .{ .grid = grid, .block = block, .shared = shared, .stream = s, .function = f, .pdl = pdl };
        for (sizes, 0..) |n, i| {
            const p: [*]const u8 = @ptrCast(args.?[i].?);
            @memcpy(got.bytes[got.used..][0..n], p[0..n]);
            got.used += n;
        }
        calls += 1;
        return 0;
    }
    fn plain(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, s: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return record(f, .{ gx, gy, gz }, .{ bx, by, bz }, shared, s, args, false);
    }
    fn extended(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        std.debug.assert(c.num_attrs == 1);
        std.debug.assert(c.attrs.?[0].id == .programmatic_stream_serialization and c.attrs.?[0].value.int == 1);
        return record(f, .{ c.grid_x, c.grid_y, c.grid_z }, .{ c.block_x, c.block_y, c.block_z }, c.shared_bytes, c.stream, args, true);
    }
    fn device(out: *cuda.abi.Device) callconv(.c) cuda.abi.Result {
        capability_calls += 1;
        out.* = 0;
        return 0;
    }
    fn attribute(out: *c_int, key: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = if (key == .compute_capability_major) major else minor;
        return 0;
    }
};
fn driver() cuda.Driver {
    var d: cuda.Driver = undefined;
    d.func_name = null;
    d.api.cuLaunchKernel = Mock.plain;
    d.api.cuLaunchKernelEx = Mock.extended;
    d.api.cuCtxGetDevice = Mock.device;
    d.api.cuDeviceGetAttribute = Mock.attribute;
    return d;
}
fn function(d: *const cuda.Driver) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(0x40) };
}
fn stream(d: *const cuda.Driver) cuda.Stream {
    return .{ .d = d, .handle = @ptrFromInt(0x80) };
}
fn same(off: Record) !void {
    try t.expect(!off.pdl and Mock.got.pdl);
    var on = Mock.got;
    on.pdl = false;
    try t.expectEqualDeep(off, on);
}
fn expected(a: *cuda.Args) void {
    Mock.sizes = a.sizes[0..a.count];
}
fn bytes(a: *const cuda.Args) !void {
    var payload: [256]u8 = @splat(0);
    var used: usize = 0;
    for (a.offsets[0..a.count], a.sizes[0..a.count]) |at, n| {
        @memcpy(payload[used..][0..n], a.storage[at..][0..n]);
        used += n;
    }
    try t.expectEqual(used, Mock.got.used);
    try t.expectEqualSlices(u8, &payload, &Mock.got.bytes);
}

test "Flash Next PDL: strict opt-in defaults off, refuses malformed values and gates sm90" {
    const name = "TF_FLASHNEXT_PDL";
    const saved = if (std.c.getenv(name)) |p| try t.allocator.dupeZ(u8, std.mem.span(p)) else null;
    defer {
        if (saved) |v| {
            _ = env.setenv(name, v, 1);
            t.allocator.free(v);
        } else {
            _ = env.unsetenv(name);
        }
    }
    const d = driver();
    Mock.capability_calls = 0;
    _ = env.unsetenv(name);
    try t.expect(!try policy.read(&d));
    _ = env.setenv(name, "0", 1);
    try t.expect(!try policy.read(&d));
    try t.expectEqual(@as(usize, 0), Mock.capability_calls);
    for ([_][]const u8{ "yes", "2", "1 ", "00" }) |bad| {
        const z = try t.allocator.dupeZ(u8, bad);
        defer t.allocator.free(z);
        _ = env.setenv(name, z, 1);
        try t.expectError(error.InvalidEnvironment, policy.read(&d));
    }
    _ = env.setenv(name, "1", 1);
    for ([_]u32{ 80, 89, 90, 100, 121 }) |cap| {
        Mock.major = @intCast(cap / 10);
        Mock.minor = @intCast(cap % 10);
        try t.expectEqual(cap >= 90, try policy.read(&d));
    }
}

test "Flash Next PDL: HC and attention preserve argument bytes and launch geometry" {
    const d = driver();
    const f = function(&d);
    const s = stream(&d);
    var g: Glue = undefined;
    g.f.add_streams = f;
    g.f.gate = f;
    var a: cuda.Args = .{};
    for ([_]u64{ 0x1000, 0x2000, 0x3000 }) |v| a.add(v);
    a.add(@as(c_int, 512));
    a.add(@as(c_int, 4));
    expected(&a);
    g.pdl = false;
    try g.addStreams(s, 0x1000, 0x2000, 0x3000, 3, 512, 4);
    const off = Mock.got;
    try bytes(&a);
    try t.expectEqual([3]u32{ 3, 2, 1 }, off.grid);
    g.pdl = true;
    try g.addStreams(s, 0x1000, 0x2000, 0x3000, 3, 512, 4);
    try same(off);
    Mock.sizes = &.{ 8, 8, 8, 8, 4, 4 };
    g.pdl = false;
    try g.attnGate(s, 1, 2, 3, 4, 2, 100, 24, 256);
    const gate = Mock.got;
    g.pdl = true;
    try g.attnGate(s, 1, 2, 3, 4, 2, 100, 24, 256);
    try same(gate);
}

test "Flash Next PDL: shared router stays plain by default, all logits shapes and topk retain bytes" {
    const d = driver();
    const f = function(&d);
    const s = stream(&d);
    var router: cuda.route_rows.RouteRows = .{ .mod = undefined, .logits_fn = f, .few_fn = f, .wide_fn = f, .topk_fn = f };
    try t.expect(!router.pdl);
    Mock.sizes = &.{ 8, 4, 8, 8, 4, 4, 4 };
    for ([_]usize{ 1, 16, 17, 64 }) |rows| {
        router.pdl = false;
        try router.logits(s, 0x1000, 256, 0x2000, 0x3000, rows, 256, 512);
        const off = Mock.got;
        router.pdl = true;
        try router.logits(s, 0x1000, 256, 0x2000, 0x3000, rows, 256, 512);
        try same(off);
    }
    Mock.sizes = &.{ 8, 4, 4, 4, 8, 8, 4, 4 };
    router.pdl = false;
    try router.topk(s, 1, 513, 512, 10, 2, 3, 11, 2, true);
    const off = Mock.got;
    router.pdl = true;
    try router.topk(s, 1, 513, 512, 10, 2, 3, 11, 2, true);
    try same(off);
    const calls = Mock.calls;
    try t.expectError(error.RowShape, router.logits(s, 1, 256, 0x2000, 3, 1, 256, 512));
    try router.logits(s, 0x1000, 256, 0x2000, 3, 0, 256, 512);
    try t.expectEqual(calls, Mock.calls);
}

test "Flash Next PDL: batch sum keeps bytes, no-op copies issue no dependent launch" {
    const d = driver();
    const s = stream(&d);
    var k: Kernels = undefined;
    k.sum = function(&d);
    Mock.sizes = &.{ 8, 8, 4, 8, 4 };
    k.pdl = false;
    try ops.sumSlices(&k, s, 1, 2, true, 257, 8);
    const off = Mock.got;
    k.pdl = true;
    try ops.sumSlices(&k, s, 1, 2, true, 257, 8);
    try same(off);
    const calls = Mock.calls;
    try ops.copyCols(&k, s, 1, 16, 0, 16, 16, 1);
    try t.expectEqual(calls, Mock.calls);
}

test "Flash Next PDL: full and half GDN fused and norm preserve bytes and shared memory" {
    const d = driver();
    const f = function(&d);
    const s = stream(&d);
    for ([_]@import("cuda_heads.zig").Heads{ .full, .half }) |h| {
        const in: gdn.In = .{ .proj = 1, .conv = 2, .cw = 3, .a_log = 4, .dt = 5, .state = 6, .state_out = 6, .norm = 7, .heads = h };
        const sizes = [_]u16{ 8, 8, 8, 8, 8, 8, 8, 4, 8, 4, 8, 8, 8, 4, 8, 4, 8 };
        Mock.sizes = sizes[0..if (h.nv == 48) 17 else 13];
        try gdn.launch(f, false, s, in, 64, 8, 9);
        const off = Mock.got;
        try gdn.launch(f, true, s, in, 64, 8, 9);
        try same(off);
        try t.expect(off.shared > 0);
        Mock.sizes = &.{ 8, 8, 8, 4, 8, 8 };
        try gdn.norm(f, false, s, 1, 2, 3, 4, 5, 17, h);
        const norm = Mock.got;
        try gdn.norm(f, true, s, 1, 2, 3, 4, 5, 17, h);
        try same(norm);
    }
}
