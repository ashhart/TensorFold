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
const Call = struct {
    how: enum { copy, gather, residual, hc },
    send: u64,
    recv: u64,
    bytes: usize,
    stream: usize,
    d: usize = 0,
    x: u64 = 0,
    xout: u64 = 0,
};

fn launchPtr(a: [*]?*anyopaque, i: usize) u64 {
    return @as(*const u64, @ptrCast(@alignCast(a[i].?))).*;
}

fn launchInt(a: [*]?*anyopaque, i: usize) u64 {
    return @as(*const u32, @ptrCast(@alignCast(a[i].?))).*;
}

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

fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, st: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
    const a = args.?;
    const how: @FieldType(Call, "how") = switch (@intFromPtr(f)) {
        1 => .gather,
        2 => .residual,
        3 => .hc,
        else => unreachable,
    };
    record(.{ .how = how, .send = launchPtr(a, 0), .recv = launchPtr(a, 1), .bytes = launchInt(a, 2) * 16, .stream = @intFromPtr(st), .d = if (how == .gather) 0 else launchPtr(a, 14), .x = if (how == .gather) 0 else launchPtr(a, 15), .xout = if (how == .gather) 0 else launchPtr(a, 16) });
    return cuda.abi.success;
}

fn driver() cuda.Driver {
    var api: cuda.abi.Api = undefined;
    api.cuMemcpyDtoDAsync_v2 = copy;
    api.cuLaunchKernel = launch;
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
const residual: cuda.roce.Post = .{ .residual_add = .{ .x = 0x2000, .xout = 0x3000 } };

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
        try std.testing.expect(!try local.resolve(rows, null));
        try expectCalls(expected(world, true, 0, &want));
        try std.testing.expect(!try local.resolve(rows, residual));
        try expectCalls(expected(world, true, 0, &want));
        for (0..world) |me| {
            const plain: Forward = .{ .d = &drv, .m = undefined, .stream = stream, .ranks = x.r[me .. me + 1], .exchange = .{ .nccl = .{ .lib = &lib, .comm = null } } };
            try std.testing.expect(!try plain.resolve(rows, null));
            try expectCalls(expected(world, false, me, &want));
            try std.testing.expect(!try plain.resolve(rows, residual));
            try expectCalls(expected(world, false, me, &want));
            // the collective the engine hands out: plain NCCL until the group is installed, then the group's route
            var g: cuda.tp_group.Group = .{ .lib = lib, .comm = null, .rank = me, .world = world };
            var b: tp.group.Backend = .{ .d = &drv, .lib = &lib, .comm = null, .rank = @intCast(me), .world = @intCast(world) };
            const grouped: Forward = .{ .d = &drv, .m = undefined, .stream = stream, .ranks = x.r[me .. me + 1], .exchange = .{ .nccl = .{ .lib = &lib, .comm = null, .collective = b.collective() } } };
            try std.testing.expect(!try grouped.resolve(rows, null));
            try expectCalls(expected(world, false, me, &want));
            try std.testing.expect(!try grouped.resolve(rows, residual));
            try expectCalls(expected(world, false, me, &want));
            b.group = &g;
            try std.testing.expect(!try grouped.resolve(rows, null));
            try expectCalls(expected(world, false, me, &want));
            try std.testing.expect(!try grouped.resolve(rows, residual));
            try expectCalls(expected(world, false, me, &want));
        }
    }
}

test "TF_GLM_ROCE_FUSE off keeps the ordinary gather; eligible TP2 posts use one exact launch" {
    const drv = driver();
    const lib = fakeLib();
    const stream: cuda.Stream = .{ .d = &drv, .handle = @ptrFromInt(stream_handle) };
    const d = (cfg.Config{}).hidden;
    const bytes = rows * d * 4;
    const hc: cuda.roce.Post = .{ .hc_post = .{ .x = 0x2000, .xout = 0x3000, .post = 0x4000, .comb = 0x5000 } };
    for (0..2) |me| {
        var x: Ranks = .{};
        x.init(2);
        var fast: cuda.roce.Gather = undefined;
        fast.d = &drv;
        fast.rank = @intCast(me);
        fast.world = 2;
        fast.layout = .{ .slot_bytes = bytes, .peers = 1, .recv = 0x100, .flags = 0x200, .send = 0x300, .ctrl = 0x400, .total = 0x500 };
        fast.region_dev = 0x6000;
        fast.counters.ptr = 0x7000;
        fast.f = .{ .d = &drv, .handle = @ptrFromInt(1) };
        fast.f_residual = .{ .d = &drv, .handle = @ptrFromInt(2) };
        fast.f_hc = .{ .d = &drv, .handle = @ptrFromInt(3) };
        fast.max_bytes = bytes;
        fast.plan = .{ .stripes = 1 };
        fast.gathers = 0;
        var g: cuda.tp_group.Group = .{ .lib = lib, .comm = null, .rank = me, .world = 2, .fast = &fast };
        var b: tp.group.Backend = .{ .d = &drv, .lib = &lib, .comm = null, .rank = @intCast(me), .world = 2, .group = &g };
        var f: Forward = .{ .d = &drv, .m = undefined, .stream = stream, .ranks = x.r[me .. me + 1], .exchange = .{ .nccl = .{ .lib = &lib, .comm = null, .collective = b.collective() } } };
        n = 0;
        const send = gathered(me) + me * bytes;
        try std.testing.expect(!try f.resolve(rows, residual));
        try expectCalls(&.{.{ .how = .gather, .send = send, .recv = gathered(me), .bytes = bytes, .stream = stream_handle }});
        f.exchange.nccl.group = &g;
        f.exchange.nccl.fuse = true;
        for ([_]cuda.roce.Post{ residual, hc }) |post| {
            const how: @FieldType(Call, "how") = if (post == .residual_add) .residual else .hc;
            try std.testing.expect(try f.resolve(rows, post));
            try expectCalls(&.{.{ .how = how, .send = send, .recv = gathered(me), .bytes = bytes, .stream = stream_handle, .d = d, .x = 0x2000, .xout = 0x3000 }});
        }
        try std.testing.expect(!try g.gatherPost(stream, send, gathered(me), bytes, d, .{ .residual_add = .{ .x = 0x2001, .xout = 0x3000 } }));
        try std.testing.expectEqual(@as(usize, 0), n); // no post or gather launched when fusion declines
    }
}
