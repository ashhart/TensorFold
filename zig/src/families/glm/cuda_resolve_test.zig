//! GLM's reduction boundary at TP 2 and 3: every copy and gather it makes, recorded through a fake driver and NCCL.
const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const fw = @import("cuda_forward.zig");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");

const Forward = fw.Forward;

/// One transport call, normalized: copies and gathers of any type compare by their bytes, pointers and stream.
const Call = struct { how: enum { copy, gather }, send: u64, recv: u64, bytes: usize, stream: usize };

var calls: [16]Call = undefined;
var n: usize = 0;

fn record(c: Call) void {
    calls[n] = c;
    n += 1;
}

fn copy(dst: cuda.abi.DevicePtr, src: cuda.abi.DevicePtr, bytes: usize, s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
    record(.{ .how = .copy, .send = src, .recv = dst, .bytes = bytes, .stream = @intFromPtr(s) });
    return cuda.abi.success;
}

fn ncclGather(s: cuda.abi.DevicePtr, r: cuda.abi.DevicePtr, count: usize, t: cuda.nccl.DataType, _: cuda.nccl.Comm, st: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
    const size: usize = switch (t) {
        .u8 => 1,
        .f32 => 4,
        else => unreachable, // the boundary moves fp32 partials, as floats or as their bytes
    };
    record(.{ .how = .gather, .send = s, .recv = r, .bytes = count * size, .stream = @intFromPtr(st) });
    return 0;
}

fn driver() cuda.Driver {
    var api: cuda.abi.Api = undefined;
    api.cuMemcpyDtoDAsync_v2 = copy;
    return .{ .lib = undefined, .api = api, .func_name = null };
}

fn fakeLib() cuda.nccl.Library {
    var api: cuda.nccl.Api = undefined;
    api.ncclAllGather = ncclGather;
    return .{ .lib = undefined, .api = api };
}

const stream_handle: usize = 0x51;
const rows = 5;

fn gathered(r: usize) u64 {
    return 0x10_0000_0000 * (r + 1); // rank r's [world][rows][hidden] fp32 buffer
}

/// What the boundary must do, worked out from the layout alone: rank r's partial sits at slot r of every buffer.
fn expected(world: usize, local: bool, me: usize, out: []Call) []Call {
    const bytes = rows * (cfg.Config{}).hidden * 4;
    var k: usize = 0;
    if (local) {
        for (0..world) |to| for (0..world) |from| {
            if (to == from) continue;
            out[k] = .{ .how = .copy, .send = gathered(from) + from * bytes, .recv = gathered(to) + from * bytes, .bytes = bytes, .stream = stream_handle };
            k += 1;
        };
    } else {
        out[0] = .{ .how = .gather, .send = gathered(me) + me * bytes, .recv = gathered(me), .bytes = bytes, .stream = stream_handle };
        k = 1;
    }
    return out[0..k];
}

const Ranks = struct {
    w: [3]weights.Weights = undefined,
    b: [3]fw.Buffers = undefined,
    r: [3]fw.Rank = undefined,

    fn init(x: *Ranks, world: usize) void {
        for (0..world) |r| {
            x.w[r].config = cfg.Config{};
            x.w[r].share = shard.Share.of(&x.w[r].config, r, world) catch unreachable;
            x.b[r].gathered = gathered(r);
            x.r[r] = .{ .w = &x.w[r], .st = undefined, .b = &x.b[r] };
        }
    }
};

fn expectCalls(want: []const Call) !void {
    try std.testing.expectEqualSlices(Call, want, calls[0..n]);
    n = 0;
}

test "the boundary's local copies, raw gathers and collective gathers (with and without the group) keep every slot, size and stream" {
    const drv = driver();
    const lib = fakeLib();
    const stream: cuda.Stream = .{ .d = &drv, .handle = @ptrFromInt(stream_handle) };
    var want: [16]Call = undefined;
    for ([_]usize{ 2, 3 }) |world| {
        var x: Ranks = .{};
        x.init(world);
        n = 0;
        const local: Forward = .{ .d = &drv, .m = undefined, .stream = stream, .ranks = x.r[0..world], .exchange = .local };
        try local.resolve(rows);
        try expectCalls(expected(world, true, 0, &want));
        for (0..world) |me| {
            const plain: Forward = .{ .d = &drv, .m = undefined, .stream = stream, .ranks = x.r[me .. me + 1], .exchange = .{ .nccl = .{ .lib = &lib, .comm = null } } };
            try plain.resolve(rows);
            try expectCalls(expected(world, false, me, &want));
            // the collective the engine hands out: plain NCCL until the group is installed, then the group's route
            var g: cuda.tp_group.Group = .{ .lib = lib, .comm = null, .rank = me, .world = world };
            var b: tp.group.Backend = .{ .d = &drv, .lib = &lib, .comm = null, .rank = @intCast(me), .world = @intCast(world) };
            const grouped: Forward = .{ .d = &drv, .m = undefined, .stream = stream, .ranks = x.r[me .. me + 1], .exchange = .{ .nccl = .{ .lib = &lib, .comm = null, .collective = b.collective() } } };
            try grouped.resolve(rows);
            try expectCalls(expected(world, false, me, &want));
            b.group = &g;
            try grouped.resolve(rows);
            try expectCalls(expected(world, false, me, &want));
        }
    }
}

var kernels: [8]struct { f: usize, gx: u32, bx: u32 } = undefined;
var kernel_count: usize = 0;

fn recordKernel(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
    kernels[kernel_count] = .{ .f = @intFromPtr(f), .gx = gx, .bx = bx };
    kernel_count += 1;
    return cuda.abi.success;
}

test "GLM fused RoCE hc_post: default keeps two launches and TP=2 opt-in removes hc_post at rows 1 and 4" {
    var drv = driver();
    drv.api.cuLaunchKernel = recordKernel;
    drv.api.cuLaunchKernelEx = @import("cuda_pdl.zig").forwardEx(recordKernel);
    const lib = fakeLib();
    const stream: cuda.Stream = .{ .d = &drv, .handle = @ptrFromInt(stream_handle) };
    var modules: fw.Modules = undefined;
    modules.index = null;
    modules.tiled = false;
    modules.packs = null;
    modules.copy_rows = modules.glue.hc_post;
    modules.dsel = false;
    modules.dscore = null;
    modules.mla = null;
    modules.tfq3 = null;
    modules.glue.hc_post = .{ .d = &drv, .handle = @ptrFromInt(0x33) };
    modules.glue.hc_post_own = modules.glue.hc_post;
    var fast: cuda.roce.Gather = undefined;
    fast.d = &drv;
    fast.rank = 0;
    fast.world = 2;
    fast.max_bytes = 65536;
    fast.layout = .of(65536, 2);
    fast.region_dev = 0x100000;
    fast.counters = undefined;
    fast.counters.ptr = 0x200000;
    fast.plan = .{};
    fast.plan.stripes = 1;
    fast.f = .{ .d = &drv, .handle = @ptrFromInt(0x21) };
    fast.hc_post = .{ .d = &drv, .handle = @ptrFromInt(0x22) };
    for ([_]usize{ 0, 1 }) |me| for ([_]usize{ 1, 4 }) |nrows| {
        var x: Ranks = .{};
        x.init(2);
        fast.rank = @intCast(me);
        x.b[me].prefill = false;
        x.b[me].x = 0x300000;
        x.b[me].post = 0x400000;
        x.b[me].comb = 0x500000;
        const f: Forward = .{ .d = &drv, .m = &modules, .stream = stream, .ranks = x.r[me .. me + 1], .exchange = .{ .nccl = .{ .lib = &lib, .comm = null, .fast = &fast } } };
        n = 0;
        kernel_count = 0;
        const default_fused = try f.resolveHcPost(nrows);
        if (!default_fused) try f.post(x.r[me], nrows);
        try std.testing.expect(!default_fused);
        try std.testing.expectEqual(@as(usize, 1), n);
        try std.testing.expect(calls[0].how == .gather);
        try std.testing.expectEqual(@as(usize, 1), kernel_count);
        try std.testing.expectEqual(@as(usize, 0x33), kernels[0].f);
        try std.testing.expectEqual(@as(u32, 128), kernels[0].bx);

        const fused: Forward = .{ .d = &drv, .m = &modules, .stream = stream, .ranks = x.r[me .. me + 1], .exchange = .{ .nccl = .{ .lib = &lib, .comm = null, .fast = &fast } }, .hc_post_fused = true };
        n = 0;
        kernel_count = 0;
        const opt_in_fused = try fused.resolveHcPost(nrows);
        if (!opt_in_fused) try fused.post(x.r[me], nrows);
        try std.testing.expect(opt_in_fused);
        try std.testing.expectEqual(@as(usize, 0), n);
        try std.testing.expectEqual(@as(usize, 1), kernel_count);
        try std.testing.expectEqual(@as(usize, 0x22), kernels[0].f);
        try std.testing.expectEqual(@as(u32, 512), kernels[0].bx);
        try std.testing.expectEqual(@as(u32, if (nrows == 1) 1 else 4), kernels[0].gx);
        kernel_count = 0;
        n = 0;
        try fused.resolve(nrows);
        try std.testing.expectEqual(@as(usize, 1), n);
        try std.testing.expect(calls[0].how == .gather);
        try std.testing.expectEqual(@as(usize, 0), kernel_count);
    };
}
