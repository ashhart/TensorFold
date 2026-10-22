//! Actual prompt projection dispatch and ownership with inert driver callbacks, without opening CUDA.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const fixture = @import("qwen35_verify_host.zig");
const T = std.testing;
const Trace = struct {
    var count: usize = 0;
    var ids: [600]usize = undefined;
    var grids: [600][3]u32 = undefined;
    var blocks: [600]u32 = undefined;
    var shares: [600]u32 = undefined;
    var args: [600][7]u64 = undefined;
    var sizes: [600][5]c_int = undefined;
    var allocations: usize = 0;
    var frees: usize = 0;
    var fail = false;
    var exceptional: u32 = 0;
    fn reset() void {
        count = 0;
        allocations = 0;
        frees = 0;
        fail = false;
        exceptional = 0;
    }
    fn alloc(out: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
        allocations += 1;
        if (fail) return cuda.abi.error_not_ready;
        out.* = 0x900000;
        return 0;
    }
    fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
        frees += 1;
        return 0;
    }
    fn download(dst: ?*anyopaque, _: u64, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        const data = @as([*]u8, @ptrCast(dst.?))[0..bytes];
        @memset(data, 0);
        if (bytes == 4) std.mem.writeInt(u32, data[0..4], exceptional, .little);
        return 0;
    }
    fn memset(_: cuda.abi.DevicePtr, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, _: c_uint, _: c_uint, shared: c_uint, _: cuda.abi.Stream, a: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const id = @intFromPtr(f);
        ids[count] = id;
        grids[count] = .{ gx, gy, gz };
        blocks[count] = bx;
        shares[count] = shared;
        if (id == 8 or id == 9) for (0..if (id == 8) @as(usize, 4) else 7) |i| {
            args[count][i] = @as(*const u64, @ptrCast(@alignCast(a.?[i].?))).*;
        };
        if (id == 9) {
            for (0..5) |i| sizes[count][i] = @as(*const c_int, @ptrCast(@alignCast(a.?[7 + i].?))).*;
        }
        count += 1;
        return 0;
    }
};
fn function(d: *cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}
fn engine(d: *cuda.Driver) q.engine.Engine {
    var e = fixture.engine(d, false);
    d.api.cuLaunchKernel = Trace.launch;
    d.api.cuMemAlloc_v2 = Trace.alloc;
    d.api.cuMemFree_v2 = Trace.free;
    d.api.cuMemcpyDtoHAsync_v2 = Trace.download;
    d.api.cuMemsetD8Async = Trace.memset;
    d.api.cuStreamSynchronize = Trace.sync;
    e.int8.kernels = .{ .quantize = function(d, 8), .prefill = function(d, 9) };
    e.wide.prefill_fns = @splat(function(d, 10));
    e.wide.gemv_fns = @splat(function(d, 11));
    e.wide.gemv_blocks = @splat(1);
    e.wide.split = null;
    e.wide.pdl = false;
    e.g = .{ .hidden = 256, .intermediate = 512, .layers = 0, .vocab = 1, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = false, .context = 1, .linear = @splat(false) };
    return e;
}
const weight: cuda.qlinear.Affine4 = .{ .w = 10, .s = 20, .b = 30, .n = 129, .k = 256, .npad = 256 };

test "prompt int8: production allocation prefix initializes optional ownership before startup" {
    const e = try q.prompt_int8.Allocation(q.engine.Engine).create(T.allocator);
    defer T.allocator.destroy(e);
    // Inspect the bytes first: a planted missing assignment must fail without reading an invalid bool or optional.
    try T.expectEqual(@as(u8, 0), std.mem.asBytes(&e.int8.on)[0]);
    try T.expect(!e.int8.on and e.int8.buf == null and e.int8.ws == 0);
    e.int8.deinit();
}

test "prompt int8: enable allocates once, failure retains no owner, and deinit is repeatable" {
    var d: cuda.Driver = undefined;
    var e = engine(&d);
    Trace.reset();
    e.int8.deinit();
    try T.expectEqual(@as(usize, 0), Trace.frees);
    Trace.fail = true;
    try T.expectError(error.NotReady, e.int8.enable(&e));
    try T.expect(!e.int8.on and e.int8.buf == null and e.int8.ws == 0);
    Trace.fail = false;
    try e.int8.enable(&e);
    try e.int8.enable(&e);
    try T.expectEqual(@as(usize, 3), Trace.allocations);
    try T.expect(e.int8.on);
    try T.expectEqual(cuda.w4a8.Kernels.scratch(q.engine.max_rows, 512), e.int8.buf.?.len);
    e.int8.deinit();
    try T.expect(!e.int8.on and e.int8.buf == null and e.int8.ws == 0);
    e.int8.deinit();
    try T.expectEqual(@as(usize, 2), Trace.frees);
    try T.expect(!e.int8.on and e.int8.buf == null and e.int8.ws == 0);
}

test "prompt int8: actual dispatch keeps option, role and affine width separate at all row edges" {
    var d: cuda.Driver = undefined;
    var e = engine(&d);
    e.int8.flag = .{ .d = &d, .ptr = 0x910000, .len = 4 };
    for ([_]bool{ false, true }) |on| for ([_]bool{ false, true }) |prompt| {
        for ([_]usize{ 1, 16, 17, 32, 127, 128, 129, 4096 }) |rows| {
            for ([_]bool{ false, true }) |other| {
                Trace.reset();
                e.int8.on = on;
                e.int8.ws = 0x900000;
                const p: q.engine.Proj = if (other) .{ .an = .{ .w = 10, .s = 20, .b = 30, .n = 129, .k = 256, .npad = 256, .bits = 8 } } else .{ .a4 = weight };
                try e.projSums(p, 0x10000, 0x20000, 0x30000, rows, prompt, 0);
                const quantized = on and prompt and !other;
                const wide_prompt = prompt and rows > 16;
                try T.expectEqual(if (quantized) (if (rows <= 16) @as(usize, 3) else 2) else if (wide_prompt) 1 else 1 + (rows + 15) / 16, Trace.count);
                if (quantized) {
                    try T.expectEqualSlices(usize, &.{ 8, 9 }, Trace.ids[0..2]);
                    try T.expectEqual([3]u32{ @intCast(rows), 1, 1 }, Trace.grids[0]);
                    try T.expectEqual(@as(u32, 128), Trace.blocks[0]);
                    try T.expectEqual(@as(u32, 256), Trace.blocks[1]);
                    try T.expectEqual(@as(u32, cuda.w4a8.smem), Trace.shares[1]);
                    try T.expectEqual(@as(u64, 0x10000), Trace.args[0][0]);
                    try T.expectEqual(@as(u64, 0x900000), Trace.args[1][0]);
                    try T.expectEqual(@as(u64, 0x900000 + rows * 256), Trace.args[1][1]);
                    try T.expectEqual(@as(u64, 0x900000 + rows * 272), Trace.args[1][2]);
                    try T.expectEqual(@as(u64, 0x30000), Trace.args[1][6]);
                    try T.expectEqual([5]c_int{ @intCast(rows), 129, 256, 256, @intCast(@min((rows + 127) / 128, 192)) }, Trace.sizes[1]);
                } else for (Trace.ids[0..Trace.count]) |id| try T.expect(id != 8 and id != 9);
            }
        }
    };
}

test "prompt int8: every prompt row takes int8, and an exceptional group is recomputed in the default precision" {
    var d: cuda.Driver = undefined;
    var e = engine(&d);
    e.int8.flag = .{ .d = &d, .ptr = 0x910000, .len = 4 };
    for ([_]usize{ 1, 16, 17, 129 }) |rows| {
        // every prompt row of an int8 request takes int8, whatever the chunk's size
        Trace.reset();
        e.int8.on = true;
        e.int8.ws = 0x900000;
        try e.projSums(.{ .a4 = weight }, 0x10000, 0x20000, 0x30000, rows, true, 0);
        try T.expectEqual(@as(usize, 8), Trace.ids[0]);
        try T.expectEqual(@as(usize, 9), Trace.ids[1]);
        // a non-finite activation or an overflowed scale: the chunk is recomputed in the default precision
        Trace.reset();
        Trace.exceptional = 1;
        try e.projSums(.{ .a4 = weight }, 0x10000, 0x20000, 0x30000, rows, true, 0);
        for (Trace.ids[0..Trace.count]) |id| try T.expect(id != 9);
        Trace.exceptional = 0;
    }
}

test "prompt int8: hand-built projection fixture initializes all newly read selection and ownership fields" {
    var driver: cuda.Driver = undefined;
    var e = fixture.engine(&driver, false);
    try T.expectEqual(@as(u8, 0), std.mem.asBytes(&e.int8.on)[0]);
    try T.expectEqual(@as(u8, 0), std.mem.asBytes(&e.affine.half_prompt)[0]);
    try T.expect(!e.int8.on and e.int8.buf == null and e.int8.flag == null and e.int8.ws == 0);
    try T.expect(!e.affine.half_prompt);
    try T.expectEqual(@as(usize, 4), @intFromPtr(e.affine.prefill22_fn.handle));
}
