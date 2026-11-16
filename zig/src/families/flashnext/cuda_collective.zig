//! One two-rank NCCL owner. Nonblocking initialization and collectives observe the control peer and a deadline.
const std = @import("std");
const cuda = @import("cuda");
const wire = @import("cuda_rank_wire.zig");
const nccl = cuda.nccl;

const Advanced = struct {
    ncclCommInitRankConfig: *const fn (*nccl.Comm, c_int, nccl.UniqueId, c_int, *nccl.Config) callconv(.c) nccl.Result,
    ncclCommAbort: *const fn (nccl.Comm) callconv(.c) nccl.Result,
    ncclCommGetAsyncError: *const fn (nccl.Comm, *nccl.Result) callconv(.c) nccl.Result,
    ncclReduceScatter: @FieldType(nccl.Api, "ncclAllReduce"),
};

pub const Collective = struct {
    lib: nccl.Library,
    api: Advanced,
    comm: nccl.Comm = null,
    link: wire.Link,
    timeout_ms: u64,
    rank: usize = 0,
    failed: bool = false,

    /// `link` has completed the source/settings handshake; it stays owned by the caller.
    pub fn init(link: wire.Link, rank: usize, timeout_ms: u64) !Collective {
        if (rank > 1 or timeout_ms == 0) return error.BadRank;
        var c: Collective = .{ .lib = try nccl.Library.open(), .api = undefined, .link = link, .timeout_ms = timeout_ms, .rank = rank };
        errdefer c.lib.close();
        try c.lib.requireSupportedVersion();
        const info = @typeInfo(Advanced).@"struct";
        inline for (info.field_names, info.field_types) |name, T| @field(c.api, name) = c.lib.lib.lookup(T, name) orelse return error.MissingSymbol;
        var diag = cuda.p2p_report.begin(rank, 2);
        defer diag.end();
        var id: nccl.UniqueId = undefined;
        const until = wire.deadline(timeout_ms);
        if (rank == 0) {
            try c.lib.check(c.lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            try link.send(std.mem.asBytes(&id), until);
        } else try link.receive(std.mem.asBytes(&id), until);
        var config: nccl.Config = .{};
        errdefer c.abort();
        try c.submitted(c.api.ncclCommInitRankConfig(&c.comm, 2, id, @intCast(rank), &config));
        try c.wait(null, until);
        cuda.p2p_report.afterInit(&diag, .{ .lib = &c.lib, .comm = c.comm }); // no driver here: no probe
        return c;
    }

    pub fn abort(c: *Collective) void {
        c.failed = true;
        if (c.comm) |comm| {
            _ = c.api.ncclCommAbort(comm);
            c.comm = null;
        }
    }

    pub fn deinit(c: *Collective) void {
        // Abort is local and also releases an idle communicator if its peer exited first.
        c.abort();
        c.lib.close();
    }

    fn submitted(c: *Collective, result: nccl.Result) !void {
        if (result == 0 or result == 7) return;
        return c.lib.check(result, "NCCL nonblocking operation");
    }

    fn wait(c: *Collective, stream: ?cuda.Stream, until: u64) !void {
        errdefer c.abort();
        while (true) {
            try c.link.checkPeer();
            if (wire.now() >= until) return error.RankTimeout;
            var state: nccl.Result = 0;
            try c.lib.check(c.api.ncclCommGetAsyncError(c.comm, &state), "ncclCommGetAsyncError");
            if (state != 0 and state != 7) return c.lib.check(state, "NCCL asynchronous operation");
            if (state == 0) {
                if (stream) |s| {
                    if (try s.done()) return;
                } else return;
            }
            wire.pause();
        }
    }

    /// Wait at a host-visible boundary while retaining peer, asynchronous-error and timeout checks.
    pub fn finish(c: *Collective, stream: cuda.Stream) !void {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        try c.wait(stream, wire.deadline(c.timeout_ms));
    }

    pub fn benchmarkShape(c: *Collective) !nccl.Shape {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        const actual = try c.lib.shape(c.comm);
        if (actual.world != 2) return error.BadRank;
        return actual;
    }

    pub fn benchmarkGather(c: *Collective, comptime T: type, d: *const cuda.Driver, stream: cuda.Stream, mine: T, out: []T) !void {
        const shape = try c.benchmarkShape();
        if (out.len != shape.world or mine.rank != shape.rank) return error.BadRank;
        const size = @sizeOf(T);
        var dev = try cuda.DeviceBuffer.alloc(d, size * out.len);
        defer dev.free();
        errdefer c.abort();
        try dev.uploadAsync(shape.rank * size, std.mem.asBytes(&mine), stream.handle);
        try c.link.checkPeer();
        try c.submitted(c.lib.api.ncclAllGather(dev.ptr + shape.rank * size, dev.ptr, size, .u8, c.comm, stream.handle));
        try c.wait(null, wire.deadline(c.timeout_ms));
        try c.finish(stream);
        try dev.download(0, std.mem.sliceAsBytes(out));
    }

    /// BF16 rank blocks are concatenated without an arithmetic reduction.
    pub fn gather(c: *Collective, stream: cuda.Stream, local: u64, out: u64, count: usize) !void {
        return c.gatherType(stream, local, out, count, .bf16);
    }

    /// FP32 rank blocks are concatenated without rounding or reduction.
    pub fn gatherF32(c: *Collective, stream: cuda.Stream, local: u64, out: u64, count: usize) !void {
        return c.gatherType(stream, local, out, count, .f32);
    }

    fn gatherType(c: *Collective, stream: cuda.Stream, local: u64, out: u64, count: usize, dtype: nccl.DataType) !void {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        if (count == 0) return;
        errdefer c.abort();
        try c.link.checkPeer();
        try c.submitted(c.lib.api.ncclAllGather(local, out, count, dtype, c.comm, stream.handle));
        try c.wait(null, wire.deadline(c.timeout_ms));
    }

    pub fn broadcast(c: *Collective, stream: cuda.Stream, values: u64, bytes: usize) !void {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        errdefer c.abort();
        try c.link.checkPeer();
        try c.submitted(c.lib.api.ncclBroadcast(values, values, bytes, .u8, 0, c.comm, stream.handle));
        try c.wait(null, wire.deadline(c.timeout_ms));
    }

    /// FP32 sum is enqueued on the compute stream; later kernels on that stream observe its result.
    pub fn sum(c: *Collective, stream: cuda.Stream, values: u64, count: usize) !void {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        if (count == 0) return;
        errdefer c.abort();
        try c.link.checkPeer();
        try c.submitted(c.lib.api.ncclAllReduce(values, values, count, .f32, .sum, c.comm, stream.handle));
        try c.wait(null, wire.deadline(c.timeout_ms));
    }

    /// sum's FP32 sum of block `rank` of `values` (blocks of `count`) alone, left in place there: a reduce-scatter.
    pub fn sumBlock(c: *Collective, stream: cuda.Stream, values: u64, count: usize) !void {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        if (count == 0) return;
        errdefer c.abort();
        try c.link.checkPeer();
        try c.submitted(c.api.ncclReduceScatter(values, values + c.rank * count * 4, count, .f32, .sum, c.comm, stream.handle));
        try c.wait(null, wire.deadline(c.timeout_ms));
    }
};
