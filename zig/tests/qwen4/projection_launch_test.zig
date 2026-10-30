//! Real backend wrappers, metadata ABI and constructor cleanup run through owned recording device tables.
const std = @import("std");
const cu = @import("cuda");
const metal = @import("metal");
const wo = @import("core").weight_only;
const C = @import("cuda_projection");
const M = @import("metal_projection");
extern fn tf_weight_only_cpu([*]const u16, [*]const u8, [*]const u8, [*]const u8, [*]u16, wo.Params, c_int, c_int) void;
const T = std.testing;
const Trace = struct {
    var live: usize = 0;
    var fail_at: ?usize = null;
    var calls: usize = 0;
    var sync_fail = false;
    var last_params: wo.Params = undefined;
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
        out.* = if (fail_at == @intFromPtr(f) - 1) 64 else 128;
        return 0;
    }
    fn sync(_: cu.abi.Stream) callconv(.c) cu.abi.Result {
        return if (sync_fail) cu.abi.error_not_ready else 0;
    }
    fn read(comptime V: type, args: ?[*]?*anyopaque, i: usize) V {
        return @as(*const V, @ptrCast(@alignCast(args.?[i].?))).*;
    }
    fn launch(f: cu.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, _: cu.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cu.abi.Result {
        const p = read(wo.Params, args, 5);
        last_params = p;
        std.debug.assert(gx == (@as(u32, @intCast(p.n)) + 3) / 4 and gy == p.rows and gz == 1 and bx == 128 and by == 1 and bz == 1 and shared == 0);
        calls += 1;
        tf_weight_only_cpu(@ptrFromInt(read(u64, args, 0)), @ptrFromInt(read(u64, args, 1)), @ptrFromInt(read(u64, args, 2)), @ptrFromInt(read(u64, args, 3)), @ptrFromInt(read(u64, args, 4)), p, @intCast(@intFromPtr(f) - 1), read(c_int, args, 6));
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
fn buf(d: *const cu.Driver, bytes: []u8) cu.DeviceBuffer {
    return .{ .d = d, .ptr = @intFromPtr(bytes.ptr), .len = bytes.len };
}
fn view(b: *const cu.DeviceBuffer) C.View {
    return .{ .buffer = b, .len = b.len };
}
fn mv(bytes: []u8) M.View {
    return .{ .buffer = .{ .bytes = bytes }, .len = bytes.len };
}
test "weight-only launch: CUDA owner ABI A16 and rejection before any launch" {
    var d = driver();
    const stream = cu.Stream.borrowed(&d, null);
    const image: [8]u8 align(8) = @splat(1);
    var k = try C.Kernels.load(&d, &image, true);
    var x: [64]u16 = @splat(0);
    x[0] = 0x3f81;
    var w: [64]u8 = @splat(0);
    w[0] = 0x38;
    var dummy: [1]u8 = @splat(0);
    var out: [1]u16 = @splat(0);
    const xb = buf(&d, std.mem.sliceAsBytes(&x));
    const wb = buf(&d, &w);
    const sb = buf(&d, &dummy);
    const ob = buf(&d, std.mem.sliceAsBytes(&out));
    const p: wo.Params = .{ .rows = 1, .n = 1, .k = 64, .format = .fp8, .group = 64, .scale_kind = .bf16 };
    try k.encode(stream, p, view(&xb), view(&wb), view(&sb), view(&sb), view(&ob));
    try T.expectEqual(@as(u16, 0x3f81), out[0]);
    try T.expectEqualDeep(p, Trace.last_params);
    const count = Trace.calls;
    try T.expectError(error.WeightOnlyAlias, k.encode(stream, p, view(&xb), view(&wb), view(&sb), view(&sb), view(&xb)));
    var bad = p;
    bad.group = 0;
    try T.expectError(error.WeightOnlyGroup, k.encode(stream, bad, view(&xb), view(&wb), view(&sb), view(&sb), view(&ob)));
    try T.expectEqual(count, Trace.calls);
    Trace.sync_fail = true;
    try T.expectError(error.NotReady, k.close(stream));
    try T.expectEqual(@as(usize, 1), Trace.live);
    Trace.sync_fail = false;
    try k.close(stream);
    try T.expectEqual(@as(usize, 0), Trace.live);
    Trace.fail_at = 2;
    try T.expectError(error.WeightOnlyThreads, C.Kernels.load(&d, &image, true));
    Trace.fail_at = null;
    try T.expectEqual(@as(usize, 0), Trace.live);
}
test "weight-only launch: Metal real encoder path and constructor prefix cleanup" {
    const a = T.allocator;
    const text = try M.source(a, @embedFile("../../kernels/metal/core/weight_only.h"), @embedFile("../../kernels/metal/core/weight_only.metal"), true);
    defer a.free(text);
    var k = try M.Kernels.load(.{}, text, true);
    var x: [64]u16 = @splat(0);
    x[0] = 0x3f81;
    var w: [64]u8 = @splat(0);
    w[0] = 0x38;
    var dummy: [1]u8 = @splat(0);
    var out: [1]u16 = @splat(0);
    const p: wo.Params = .{ .rows = 1, .n = 1, .k = 64, .format = .fp8, .group = 64, .scale_kind = .bf16 };
    const xv = mv(std.mem.sliceAsBytes(&x));
    const wv = mv(&w);
    const sv = mv(&dummy);
    const ov = mv(std.mem.sliceAsBytes(&out));
    try k.encode(.{}, p, xv, wv, sv, sv, ov);
    try T.expectEqual(@as(u16, 0x3f81), out[0]);
    try T.expectEqual(@as(usize, 128), metal.threads.width);
    const count = metal.calls;
    try T.expectError(error.WeightOnlyAlias, k.encode(.{}, p, xv, wv, sv, sv, xv));
    try T.expectError(error.WeightOnlyContext, k.encode(.{ .owner = 2 }, p, xv, wv, sv, sv, ov));
    try T.expectEqual(count, metal.calls);
    k.deinit();
    try T.expectEqual(@as(usize, 0), metal.live);
    metal.fail_pipe = 2;
    try T.expectError(error.PlantedPipelineFailure, M.Kernels.load(.{}, text, true));
    metal.fail_pipe = null;
    try T.expectEqual(@as(usize, 0), metal.live);
}
