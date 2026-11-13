//! The exchange's rows and calls over a faked NCCL and driver, and its reduce-scatter's failures against the sum's.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const batch = @import("cuda_batch.zig");
const chunk = @import("cuda_chunk.zig");
const mtp = @import("cuda_mtp.zig");
const exchange = @import("cuda_moe_exchange.zig");
const Collective = @import("cuda_collective.zig").Collective;
const nccl = cuda.nccl;
const abi = cuda.abi;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

const dims = wt.dims;
const slots = wt.slots;
const row_y = slots * dims * 4;

// Each NCCL call and launch in order; an asynchronous state the waits read.
const Fake = struct {
    const Call = struct { kind: enum { scatter, gather, reduce, launch }, send: u64 = 0, recv: u64 = 0, count: usize = 0, f32: bool = true, grid: [6]u32 = @splat(0), args: [5]u64 = @splat(0) };
    var calls: [4]Call = undefined;
    var len: usize = 0;
    var state: nccl.Result = 0;
    var aborts: usize = 0;

    fn reset() void {
        len = 0;
        state = 0;
        aborts = 0;
    }
    fn record(c: Call) void {
        calls[len] = c;
        len += 1;
    }
    fn scatter(send: abi.DevicePtr, recv: abi.DevicePtr, count: usize, t: nccl.DataType, op: nccl.RedOp, _: nccl.Comm, _: abi.Stream) callconv(.c) nccl.Result {
        record(.{ .kind = .scatter, .send = send, .recv = recv, .count = count, .f32 = t == .f32 and op == .sum });
        return 0;
    }
    fn reduce(send: abi.DevicePtr, recv: abi.DevicePtr, count: usize, t: nccl.DataType, op: nccl.RedOp, _: nccl.Comm, _: abi.Stream) callconv(.c) nccl.Result {
        record(.{ .kind = .reduce, .send = send, .recv = recv, .count = count, .f32 = t == .f32 and op == .sum });
        return 0;
    }
    fn gather(send: abi.DevicePtr, recv: abi.DevicePtr, count: usize, t: nccl.DataType, _: nccl.Comm, _: abi.Stream) callconv(.c) nccl.Result {
        record(.{ .kind = .gather, .send = send, .recv = recv, .count = count, .f32 = t == .f32 });
        return 0;
    }
    fn read(comptime T: type, args: ?[*]?*anyopaque, i: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args.?[i].?))).*;
    }
    fn launch(_: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, _: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        var c: Call = .{ .kind = .launch, .grid = .{ gx, gy, gz, bx, by, bz } };
        for (0..3) |i| c.args[i] = read(u64, args, i);
        for (3..5) |i| c.args[i] = @intCast(read(c_int, args, i));
        record(c);
        return 0;
    }
    fn asyncError(_: nccl.Comm, out: *nccl.Result) callconv(.c) nccl.Result {
        out.* = state;
        return 0;
    }
    fn abort(_: nccl.Comm) callconv(.c) nccl.Result {
        aborts += 1;
        return 0;
    }
};

const Bufs = struct { y: u64 = 0x1000_0000, wts: u64 = 0x2000_0000, moe_f32: u64 = 0x3000_0000 };

const Pass = struct {
    s: *const Bufs,
    k: struct { moe_sum: cuda.Function },
    stream: cuda.Stream,
    pending_f32: bool = false,
};

/// Rank `rank`'s collective over one end of `pair`, its NCCL faked.
fn collective(pair: *[2]c_int, rank: usize, timeout_ms: u64) !Collective {
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, pair) != 0) return error.SocketPair;
    var c: Collective = undefined;
    c.failed = false;
    c.release = null;
    c.comm = @ptrFromInt(0x1000);
    c.parent = null;
    c.link = .{ .fd = pair[0] };
    c.timeout_ms = timeout_ms;
    c.rank = rank;
    c.lib.api.ncclAllReduce = Fake.reduce;
    c.lib.api.ncclAllGather = Fake.gather;
    c.api.ncclReduceScatter = Fake.scatter;
    c.api.ncclCommGetAsyncError = Fake.asyncError;
    c.api.ncclCommAbort = Fake.abort;
    return c;
}

fn close(pair: [2]c_int) void {
    for (pair) |fd| if (fd >= 0) {
        _ = std.c.close(fd);
    };
}

test "rounds and windows keep the all-reduce; a wider prompt pass splits, at --tp 2 only" {
    for (1..mtp.round_rows + 1) |n| {
        try expect(!exchange.rowSplit(2, n));
        try expectEqual(n, exchange.clearedRows(2, n));
    }
    for ([_]usize{ mtp.round_rows + 1, batch.default_chunk, chunk.lone_rows, batch.max_chunk }) |n| {
        try expect(exchange.rowSplit(2, n));
        try expect(!exchange.rowSplit(1, n));
        try expectEqual(n, exchange.clearedRows(1, n));
    }
}

test "the ranks' rows cover a split pass once, in blocks that pad at most the scratch's one spare row" {
    var n: usize = mtp.round_rows + 1;
    while (n <= batch.max_chunk) : (n += if (n < 600) 1 else 997) {
        const h = exchange.block(n);
        const r0 = cuda.rank_scatter.share(n, 2, 0);
        const r1 = cuda.rank_scatter.share(n, 2, 1);
        try expect(r0[0] == 0 and r0[1] == h and r1[0] == h and r1[1] == n);
        try expect(exchange.clearedRows(2, n) == 2 * h and 2 * h - n <= 1);
    }
    try expectEqual(@as(usize, 2 * 8192), exchange.clearedRows(2, batch.max_chunk));
}

test "a split pass reduce-scatters its rank's block in place, sums that block's rows and gathers the sums in place" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const bufs: Bufs = .{};
    for ([_]usize{ 0, 1 }) |rank| for ([_]usize{ 65, 66, 1537, batch.default_chunk, batch.max_chunk }) |n| {
        var pair: [2]c_int = .{ -1, -1 };
        var c = try collective(&pair, rank, 1000);
        defer close(pair);
        var pass: Pass = .{ .s = &bufs, .k = .{ .moe_sum = .{ .d = &d, .handle = @ptrFromInt(1) } }, .stream = .{ .d = &d, .handle = null } };
        Fake.reset();
        try exchange.run(&pass, &c, n);
        const h = (n + 1) / 2;
        const lo = rank * h;
        const rows = if (rank == 0) h else n - h;
        try expectEqual(@as(usize, 3), Fake.len);
        const rs = Fake.calls[0];
        try expect(rs.kind == .scatter and rs.f32 and rs.send == bufs.y and rs.recv == bufs.y + lo * row_y and rs.count == h * slots * dims);
        const k = Fake.calls[1];
        try expect(k.kind == .launch);
        try expectEqual([6]u32{ @intCast(rows), dims / 256, 1, 256, 1, 1 }, k.grid);
        try expectEqual([5]u64{ bufs.y + lo * row_y, bufs.wts + lo * slots * 4, bufs.moe_f32 + lo * dims * 4, dims, slots }, k.args);
        const ag = Fake.calls[2];
        try expect(ag.kind == .gather and ag.f32 and ag.send == bufs.moe_f32 + rank * h * dims * 4 and ag.recv == bufs.moe_f32 and ag.count == h * dims);
        try expect(pass.pending_f32 and !c.failed and Fake.aborts == 0);
    };
}

const Outcome = struct { err: ?anyerror, calls: usize, aborted: bool, refused: bool };

/// sum or sumBlock under one failure: what it returned, what reached NCCL, and whether it left the communicator dead.
fn outcome(comptime block: bool, failure: enum { none, empty, peer, async_error, stuck }) !Outcome {
    var pair: [2]c_int = .{ -1, -1 };
    var c = try collective(&pair, 1, 50);
    defer close(pair);
    var d: cuda.Driver = undefined;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    Fake.reset();
    switch (failure) {
        .peer => {
            _ = std.c.close(pair[1]);
            pair[1] = -1;
        },
        .async_error => Fake.state = 3,
        .stuck => Fake.state = nccl.in_progress,
        else => {},
    }
    const count: usize = if (failure == .empty) 0 else 4099;
    const f = if (block) Collective.sumBlock else Collective.sum;
    var err: ?anyerror = null;
    f(&c, stream, 0x5000_0000, count) catch |e| {
        err = e;
    };
    const calls = Fake.len;
    const refused = if (f(&c, stream, 0x5000_0000, 1)) |_| false else |e| e == error.RankDisconnected;
    return .{ .err = err, .calls = calls, .aborted = c.failed and c.comm == null and Fake.aborts == 1, .refused = refused };
}

test "the reduce-scatter keeps the sum's contract: a dead peer, an asynchronous error or the deadline aborts it for good" {
    const cases = .{ .none, .empty, .peer, .async_error, .stuck };
    inline for (cases) |failure| {
        const sum = try outcome(false, failure);
        const scatter = try outcome(true, failure);
        try expectEqual(sum, scatter);
    }
    try expectEqual(Outcome{ .err = null, .calls = 1, .aborted = false, .refused = false }, try outcome(true, .none));
    try expectEqual(Outcome{ .err = null, .calls = 0, .aborted = false, .refused = false }, try outcome(true, .empty));
    try expectEqual(Outcome{ .err = error.RankDisconnected, .calls = 0, .aborted = true, .refused = true }, try outcome(true, .peer));
    try expectEqual(Outcome{ .err = error.NcclFailed, .calls = 1, .aborted = true, .refused = true }, try outcome(true, .async_error));
    try expectEqual(Outcome{ .err = error.RankTimeout, .calls = 1, .aborted = true, .refused = true }, try outcome(true, .stuck));
}
