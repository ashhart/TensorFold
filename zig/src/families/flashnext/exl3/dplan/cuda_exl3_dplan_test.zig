//! Host tests of the EXL3 device planner: the demand trace, its ownership checkpoints and the planner's state words.

const std = @import("std");
const cuda = @import("cuda");
const pool_ = @import("../offload/cuda_exl3_pool.zig");
const layout = @import("exl3_dplan_layout.zig");
const dplan = @import("cuda_exl3_dplan.zig");
const DevPlan = dplan.DevPlan;
const PoolState = dplan.PoolState;
const demand_trace = dplan.demand_trace;
const stateWords = dplan.stateWords;
const experts_n = layout.experts_n;
const recent_window = layout.recent_window;
const recent_words_per_row = experts_n / 32;
const misc_words = layout.misc_words;
const recentOffset = layout.recentOffset;
const miscOffset = layout.miscOffset;

test "exl3: demand trace diagnostic APIs reject disabled capture" {
    var dp: DevPlan = undefined;
    dp.trace_mem = null;
    try std.testing.expectEqual(@as(usize, 0), dp.traceDeviceBytes());
    try std.testing.expectError(error.TraceDisabled, dp.readTraceAfterConsumers(0, &.{}));
    try std.testing.expectError(error.TraceDisabled, dp.downloadTraceAfterConsumers(std.testing.allocator, 0));
    try dp.traceOwnership(undefined, true);
    try dp.traceOwnership(undefined, false);
}

test "exl3: ownership trace checkpoints use ragged pool state and serialized compute stream" {
    const Probe = struct {
        var calls: usize = 0;
        var arguments: [4][4]u64 = undefined;
        var streams: [4]cuda.abi.Stream = undefined;

        fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, stream: cuda.abi.Stream, parameters: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
            inline for (.{ u64, c_int, u64, c_int }, 0..) |Type, index| {
                const value: *const Type = @ptrCast(@alignCast(parameters.?[index].?));
                arguments[calls][index] = @intCast(value.*);
            }
            streams[calls] = stream;
            calls += 1;
            return 0;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Probe.launch;
    var entries = [_]PoolState{ try PoolState.init(10, 0), try PoolState.init(31, stateWords(10)) };
    var device: DevPlan = undefined;
    device.pool_states = &entries;
    device.state.ptr = 0x10000;
    device.trace_mem = .{ .d = &driver, .ptr = 0x100000, .len = 2 * demand_trace.pool_bytes, .borrowed = true };
    device.trace_checkpoint_fn = .{ .d = &driver, .handle = @ptrFromInt(1) };
    const stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(2) };
    Probe.calls = 0;
    try device.traceOwnership(stream, true);
    try device.traceOwnership(stream, false);
    for (0..4) |call| {
        const index = call % 2;
        try std.testing.expectEqual([4]u64{ device.state.ptr + entries[index].byteOffset(), entries[index].lines, 0x100000 + index * demand_trace.pool_bytes, @intFromBool(call < 2) }, Probe.arguments[call]);
        try std.testing.expectEqual(stream.handle, Probe.streams[call]);
    }
    device.trace_mem = null;
    device.trace_checkpoint_fn = null;
    try device.traceOwnership(stream, true);
    try std.testing.expectEqual(@as(usize, 4), Probe.calls);
}

test "exl3: recent-demand planner state is separate from lifetime profile counts" {
    try std.testing.expectEqual(@as(usize, 3 * experts_n), recentOffset(10) - 20);
    try std.testing.expectEqual(@as(usize, recent_window * recent_words_per_row), miscOffset(10) - recentOffset(10));
    try std.testing.expectEqual(@as(usize, misc_words), stateWords(10) - miscOffset(10));
}

test "exl3: demand trace planner arguments stay fixed and joined reads isolate pools" {
    const Probe = struct {
        var trace_address: u64 = 0;
        var launches: usize = 0;

        fn launch(function: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, parameters: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
            launches += 1;
            if (@intFromPtr(function.?) == 1) {
                const address: *const u64 = @ptrCast(@alignCast(parameters.?[25].?));
                trace_address = address.*;
            }
            return 0;
        }

        fn download(destination: ?*anyopaque, source: u64, length: usize) callconv(.c) c_int {
            const target: [*]u8 = @ptrCast(destination.?);
            const origin: [*]const u8 = @ptrFromInt(source);
            @memcpy(target[0..length], origin[0..length]);
            return 0;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Probe.launch;
    driver.api.cuMemcpyDtoH_v2 = Probe.download;
    const words = try std.testing.allocator.alloc(u32, 2 * demand_trace.pool_words);
    defer std.testing.allocator.free(words);
    for (0..2) |index| @memcpy(words[index * demand_trace.pool_words ..][0..demand_trace.header_words], &try demand_trace.header(index, index, 10, 16));
    var resident: pool_.Pool = undefined;
    resident.lines.tables.ptr = 4096;
    resident.lines.mem.ptr = 8192;
    resident.lines.tlay = cuda.exl3_pool.Tables.init(513);
    resident.lines.layout = cuda.exl3_pool.Layout.init(.{ 256, 256, 256 });
    var pools = [_]*pool_.Pool{ &resident, &resident };
    var states = [_]PoolState{ try PoolState.init(10, 0), try PoolState.init(10, stateWords(10)) };
    var device: DevPlan = undefined;
    device.pools = &pools;
    device.pool_states = &states;
    device.max_rows = 16;
    device.state.ptr = 16384;
    device.statics.ptr = 32768;
    device.work.ptr = 65536;
    device.plan_fn = .{ .d = &driver, .handle = @ptrFromInt(1) };
    device.copy_fn = .{ .d = &driver, .handle = @ptrFromInt(2) };
    device.copy_blocks = 1;
    device.overlap = false;
    device.trace_mem = .{ .d = &driver, .ptr = @intFromPtr(words.ptr), .len = words.len * 4, .borrowed = true };
    const out = try std.testing.allocator.alloc(u32, demand_trace.pool_words);
    defer std.testing.allocator.free(out);
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    Probe.launches = 0;
    for (0..2) |index| {
        for (0..2) |_| {
            try device.plan(stream, index, 1024, 1);
            try std.testing.expectEqual(@intFromPtr(words.ptr) + index * demand_trace.pool_bytes, Probe.trace_address);
        }
        const view = try device.readTraceAfterConsumers(index, out);
        try std.testing.expectEqual(@as(usize, 0), view.recorded());
        try std.testing.expectEqual(@as(u32, @intCast(index)), view.words[2]);
    }
    try std.testing.expectEqual(@as(usize, 8), Probe.launches);
    try std.testing.expectEqual(words.len * 4, device.traceDeviceBytes());
    try std.testing.expectError(error.Invalid, device.readTraceAfterConsumers(2, out));
    try std.testing.expectError(error.Invalid, device.readTraceAfterConsumers(0, out[1..]));
    device.trace_mem = null;
    try device.plan(stream, 0, 1024, 1);
    try std.testing.expectEqual(@as(u64, 0), Probe.trace_address);
    try std.testing.expectEqual(@as(usize, 0), device.traceDeviceBytes());
    try std.testing.expectError(error.TraceDisabled, device.readTraceAfterConsumers(0, out));
}
