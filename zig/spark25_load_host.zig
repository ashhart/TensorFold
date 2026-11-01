//! The actual Spark kernel constructor over inert images and a driver that records every load boundary.
const std = @import("std");
const cuda = @import("cuda");
const spark = @import("spark25");
var modules: usize = 0;
var closed: usize = 0;
var functions: usize = 0;
var attributes: usize = 0;
var prompt128: usize = 0;
var trace: std.crypto.hash.sha2.Sha256 = undefined;
fn reset() void {
    modules = 0;
    closed = 0;
    functions = 0;
    attributes = 0;
    prompt128 = 0;
    trace = .init(.{});
}
fn number(n: usize) void {
    const v: u64 = n;
    trace.update(std.mem.asBytes(&v));
}
fn moduleLoad(out: *cuda.abi.Module, _: ?*const anyopaque) callconv(.c) cuda.abi.Result {
    modules += 1;
    out.* = @ptrFromInt(modules);
    trace.update("L");
    number(modules);
    return 0;
}
fn moduleClose(m: cuda.abi.Module) callconv(.c) cuda.abi.Result {
    closed += 1;
    trace.update("U");
    number(@intFromPtr(m));
    return 0;
}
fn functionGet(out: *cuda.abi.Function, m: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
    const s = std.mem.span(name);
    if (std.mem.eql(u8, s, "tf_window_prompt_128_h4_pg") or std.mem.eql(u8, s, "tf_window_prompt_128_h8_pg")) prompt128 += 1;
    functions += 1;
    out.* = @ptrFromInt(1000 + functions);
    trace.update("F");
    number(@intFromPtr(m));
    trace.update(s);
    trace.update(&.{0});
    return 0;
}
fn attrSet(f: cuda.abi.Function, a: cuda.abi.FunctionAttribute, value: c_int) callconv(.c) cuda.abi.Result {
    attributes += 1;
    trace.update("A");
    number(@intFromPtr(f));
    number(@intCast(@backingInt(a)));
    number(@intCast(value));
    return 0;
}
fn occupancy(out: *c_int, _: cuda.abi.Function, _: c_int, _: usize) callconv(.c) cuda.abi.Result {
    out.* = 1;
    return 0;
}
fn deviceAttr(out: *c_int, a: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
    out.* = switch (a) {
        .multiprocessor_count => 32,
        .compute_capability_major => 12,
        .compute_capability_minor => 1,
        else => 0,
    };
    return 0;
}
fn currentDevice(out: *cuda.abi.Device) callconv(.c) cuda.abi.Result {
    out.* = 0;
    return 0;
}
fn driver() cuda.Driver {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuModuleLoadData = moduleLoad;
    d.api.cuModuleUnload = moduleClose;
    d.api.cuModuleGetFunction = functionGet;
    d.api.cuFuncSetAttribute = attrSet;
    d.api.cuOccupancyMaxActiveBlocksPerMultiprocessor = occupancy;
    d.api.cuDeviceGetAttribute = deviceAttr;
    d.api.cuCtxGetDevice = currentDevice;
    return d;
}
test "Spark load: default constructor never touches the new 128 page kernels" {
    reset();
    var d = driver();
    const ctx: cuda.Context = .{ .d = &d, .device = 0, .handle = null };
    var k = try spark.kernels.Kernels.load(&ctx);
    errdefer k.deinit();
    try std.testing.expect(k.win.pages != null);
    try std.testing.expectEqual(@as(usize, 0), prompt128);
    k.deinit();
    try std.testing.expectEqual(modules, closed);
    std.debug.print("SPARK_DEFAULT_LOAD modules={d} functions={d} attributes={d} closed={d} trace={x}\n", .{ modules, functions, attributes, closed, trace.finalResult() });
}

test "Spark load: explicit 128 page opt-in resolves both new functions exactly once" {
    reset();
    var d = driver();
    const ctx: cuda.Context = .{ .d = &d, .device = 0, .handle = null };
    var k = try spark.kernels.Kernels.load(&ctx);
    defer k.deinit();
    try std.testing.expectEqual(@as(usize, 0), prompt128);
    const previous_functions = functions;
    const previous_modules = modules;
    const previous_attributes = attributes;
    try k.win.loadPromptPages128(&d);
    try std.testing.expectEqual(@as(usize, 2), prompt128);
    try std.testing.expectEqual(previous_functions + 2, functions);
    try std.testing.expectEqual(previous_attributes + 2, attributes);
    try std.testing.expectEqual(previous_modules, modules);
    try std.testing.expect(k.win.prompt_pages128[0] != null and k.win.prompt_pages128[1] != null);
    try k.win.loadPromptPages128(&d);
    try k.win.loadPages(&d);
    try std.testing.expectEqual(previous_functions + 2, functions);
    try std.testing.expectEqual(previous_attributes + 2, attributes);
    try std.testing.expectEqual(previous_modules, modules);
}
