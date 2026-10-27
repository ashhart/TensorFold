//! One two-rank NCCL owner. Nonblocking initialization and collectives observe the control peer and a deadline.
const std = @import("std");
const cuda = @import("cuda");
const wire = @import("cuda_rank_wire.zig");
const nccl = cuda.nccl;

const Advanced = struct {
    ncclCommInitRankConfig: *const fn (*nccl.Comm, c_int, nccl.UniqueId, c_int, *nccl.Config) callconv(.c) nccl.Result,
    ncclCommAbort: *const fn (nccl.Comm) callconv(.c) nccl.Result,
    ncclCommGetAsyncError: *const fn (nccl.Comm, *nccl.Result) callconv(.c) nccl.Result,
};

/// How long a wait polls before it sleeps between polls (a decode step's work ends within a few milliseconds).
const spin_ns: u64 = 3 * std.time.ns_per_ms;

pub const Collective = struct {
    lib: nccl.Library,
    api: Advanced,
    comm: nccl.Comm = null,
    link: wire.Link,
    timeout_ms: u64,
    failed: bool = false,

    /// `link` has completed the source/settings handshake; it stays owned by the caller.
    pub fn init(link: wire.Link, rank: usize, timeout_ms: u64) !Collective {
        if (rank > 1 or timeout_ms == 0) return error.BadRank;
        var c: Collective = .{ .lib = try nccl.Library.open(), .api = undefined, .link = link, .timeout_ms = timeout_ms };
        errdefer c.lib.close();
        try c.lib.requireSupportedVersion();
        const info = @typeInfo(Advanced).@"struct";
        inline for (info.field_names, info.field_types) |name, T| @field(c.api, name) = c.lib.lib.lookup(T, name) orelse return error.MissingSymbol;
        var id: nccl.UniqueId = undefined;
        const until = wire.deadline(timeout_ms);
        if (rank == 0) {
            try c.lib.check(c.lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            try link.send(std.mem.asBytes(&id), until);
        } else try link.receive(std.mem.asBytes(&id), until);
        var config: nccl.Config = .{};
        // TF_FLASHNEXT_NCCL_BLOCKING=1: a blocking communicator (NCCL enqueues on the calling thread; a nonblocking
        if (std.c.getenv("TF_FLASHNEXT_NCCL_BLOCKING")) |v| if (std.mem.eql(u8, std.mem.span(v), "1")) {
            config.blocking = 1;
        };
        errdefer c.abort();
        try c.submitted(c.api.ncclCommInitRankConfig(&c.comm, 2, id, @intCast(rank), &config));
        try c.wait(null, until);
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
        var polls: u32 = 0;
        var spin_until: u64 = 0;
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
            // a nonblocking enqueue lands within tens of microseconds: poll that long before sleeping.
            polls += 1;
            if (polls == 1) spin_until = wire.now() + spin_ns;
            if (wire.now() < spin_until) std.atomic.spinLoopHint() else wire.pause();
        }
    }

    /// Wait at a host-visible boundary while retaining peer, asynchronous-error and timeout checks.
    pub fn finish(c: *Collective, stream: cuda.Stream) !void {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        try c.wait(stream, wire.deadline(c.timeout_ms));
    }

    /// Disjoint BF16 vocabulary columns are concatenated without an arithmetic reduction.
    pub fn gather(c: *Collective, stream: cuda.Stream, local: u64, out: u64, count: usize) !void {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        if (count == 0) return;
        errdefer c.abort();
        try c.link.checkPeer();
        try c.submitted(c.lib.api.ncclAllGather(local, out, count, .bf16, c.comm, stream.handle));
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
};
