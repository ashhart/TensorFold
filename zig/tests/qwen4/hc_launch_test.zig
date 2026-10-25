//! Real HC adapters execute their scalar source body through owned recording Metal and CUDA boundaries.
const std = @import("std");
const cu = @import("cuda");
const metal = @import("metal");
const core = @import("core");
const h = core.hc_activation;
const C = core.hc_activation_cuda_adapter(h, cu);
const M = core.hc_activation_metal_adapter(h, metal);
extern fn tf_hc_activation_cpu([*]const u16, [*]u16, c_int, c_int, c_int, c_int, c_int) void;
const T = std.testing;
const Trace = struct {
    var live: usize = 0;
    var calls: usize = 0;
    var fail_at: ?usize = null;
    var sync_fail = false;
    fn load(out: *cu.abi.Module, _: ?*const anyopaque) callconv(.c) cu.abi.Result {
        live += 1;
        out.* = @ptrFromInt(100);
        return 0;
    }
    fn unload(_: cu.abi.Module) callconv(.c) cu.abi.Result {
        live -= 1;
        return 0;
    }
    fn function(out: *cu.abi.Function, _: cu.abi.Module, name: [*:0]const u8) callconv(.c) cu.abi.Result {
        for (C.names, 0..) |n, i| if (std.mem.eql(u8, n, std.mem.span(name))) {
            out.* = @ptrFromInt(i + 1);
            return 0;
        };
        return cu.abi.error_not_found;
    }
    fn attribute(out: *c_int, _: cu.abi.FunctionAttribute, f: cu.abi.Function) callconv(.c) cu.abi.Result {
        out.* = if (fail_at == @intFromPtr(f) - 1) 128 else 256;
        return 0;
    }
    fn sync(_: cu.abi.Stream) callconv(.c) cu.abi.Result {
        return if (sync_fail) cu.abi.error_not_ready else 0;
    }
    fn read(comptime V: type, args: ?[*]?*anyopaque, i: usize) V {
        return @as(*const V, @ptrCast(@alignCast(args.?[i].?))).*;
    }
    fn launch(f: cu.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, _: cu.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cu.abi.Result {
        const rows = read(c_int, args, 2);
        const cols = read(c_int, args, 3);
        std.debug.assert(gx == (@as(u32, @intCast(rows * cols)) + 255) / 256 and gy == 1 and gz == 1 and bx == 256 and by == 1 and bz == 1 and shared == 0);
        calls += 1;
        tf_hc_activation_cpu(@ptrFromInt(read(u64, args, 0)), @ptrFromInt(read(u64, args, 1)), rows * cols, read(c_int, args, 4), read(c_int, args, 5), @intCast(@intFromPtr(f) - 1), read(c_int, args, 6));
        return 0;
    }
};
fn driver() cu.Driver {
    var d: cu.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuModuleLoadData = Trace.load;
    d.api.cuModuleUnload = Trace.unload;
    d.api.cuModuleGetFunction = Trace.function;
    d.api.cuFuncGetAttribute = Trace.attribute;
    d.api.cuStreamSynchronize = Trace.sync;
    d.api.cuLaunchKernel = Trace.launch;
    return d;
}
fn cb(d: *const cu.Driver, b: []u16) cu.DeviceBuffer {
    return .{ .d = d, .ptr = @intFromPtr(b.ptr), .len = b.len * 2 };
}
fn cv(b: *const cu.DeviceBuffer) C.View {
    return .{ .buffer = b, .len = b.len };
}
fn mv(b: []u16) M.View {
    return .{ .buffer = .{ .bytes = std.mem.sliceAsBytes(b) }, .len = b.len * 2 };
}
test "HC launch: CUDA ABI, no writes on refusal, module retention and prefix cleanup" {
    var d = driver();
    const s = cu.Stream.borrowed(&d, null);
    const image: [8]u8 align(8) = @splat(1);
    var k = try C.Kernels.load(&d, &image, true);
    var x: [257]u16 = @splat(0x3f80);
    var y: [257]u16 = @splat(0xffff);
    const xb = cb(&d, &x);
    const yb = cb(&d, &y);
    const p: h.Params = .{ .rows = 1, .cols = 257, .streams = 4, .injection = 1 };
    try k.encode(s, p, cv(&xb), cv(&yb));
    for (y) |v| try T.expectEqual(@as(u16, 0x3f90), v);
    const n = Trace.calls;
    var overlap = xb;
    overlap.ptr += 2;
    overlap.len -= 2;
    try T.expectError(error.HcActivationBuffer, k.encode(s, p, cv(&xb), cv(&overlap)));
    const short: h.Params = .{ .rows = 1, .cols = 256, .streams = 4, .injection = 1 };
    try T.expectError(error.HcActivationAlias, k.encode(s, short, cv(&xb), cv(&overlap)));
    var foreign = d;
    try T.expectError(error.HcActivationContext, k.encode(cu.Stream.borrowed(&foreign, null), p, cv(&xb), cv(&yb)));
    try T.expectEqual(n, Trace.calls);
    Trace.sync_fail = true;
    try T.expectError(error.NotReady, k.close(s));
    try T.expectEqual(@as(usize, 1), Trace.live);
    Trace.sync_fail = false;
    try k.close(s);
    for (0..4) |i| {
        Trace.fail_at = i;
        try T.expectError(error.HcActivationThreads, C.Kernels.load(&d, &image, true));
        try T.expectEqual(@as(usize, 0), Trace.live);
    }
    Trace.fail_at = null;
    k = try C.Kernels.load(&d, &image, false);
    defer k.close(s) catch unreachable;
    try T.expectError(error.HcActivationControl, k.testEncode(s, p, cv(&xb), cv(&yb), .tail, 0));
}
test "HC launch: real Metal bindings, exact aliases, context refusal and constructor prefix cleanup" {
    const text = try M.source(T.allocator, @embedFile("../../kernels/metal/core/weight_only.h"), @embedFile("../../kernels/metal/core/hc_activation.h"), @embedFile("../../kernels/metal/core/hc_activation.metal"), true);
    defer T.allocator.free(text);
    var k = try M.Kernels.load(.{}, text, true);
    var x: [257]u16 = @splat(0x3f80);
    var y: [257]u16 = @splat(0xffff);
    const p: h.Params = .{ .rows = 1, .cols = 257, .streams = 4, .injection = 1 };
    try k.encode(.{}, p, mv(&x), mv(&y));
    for (y) |v| try T.expectEqual(@as(u16, 0x3f90), v);
    try T.expectEqual(@as(usize, 256), metal.threads.width);
    try T.expectEqual(@as(usize, 2), metal.groups.width);
    const n = metal.calls;
    try T.expectError(error.HcActivationContext, k.encode(.{ .owner = 2 }, p, mv(&x), mv(&y)));
    try T.expectEqual(n, metal.calls);
    const short: h.Params = .{ .rows = 1, .cols = 256, .streams = 4, .injection = 0 };
    try T.expectError(error.HcActivationAlias, k.encode(.{}, short, mv(&x), mv(x[1..])));
    try k.encode(.{}, p, mv(&x), mv(&x));
    k.deinit();
    try T.expectEqual(@as(usize, 0), metal.live);
    for (0..4) |i| {
        metal.fail_pipe = i;
        try T.expectError(error.PlantedPipelineFailure, M.Kernels.load(.{}, text, true));
        try T.expectEqual(@as(usize, 0), metal.live);
    }
    metal.fail_pipe = null;
    try T.expectError(error.HcActivationSource, M.source(T.allocator, "", "", "", false));
}
