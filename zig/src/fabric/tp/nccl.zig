//! The NCCL backend: one communicator a rank, every collective one enqueue on the caller's stream.
const std = @import("std");
const cuda = @import("cuda");
const collective = @import("collective.zig");
const nccl = cuda.nccl;
const DevicePtr = collective.DevicePtr;
const DType = collective.DType;
const Op = collective.Op;
const Error = collective.Error;

pub fn dataType(t: DType) nccl.DataType {
    return switch (t) {
        .u8 => .u8,
        .i32 => .i32,
        .i64 => .i64,
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
    };
}

pub fn redOp(op: Op) nccl.RedOp {
    return switch (op) {
        .sum => .sum,
        .max => .max,
        .min => .min,
    };
}

pub const Nccl = struct {
    lib: *const nccl.Library,
    d: *const cuda.Driver,
    comm: nccl.Comm,
    me: u32,
    size: u32,
    aborted: std.atomic.Value(bool) = .init(false),
    /// The barrier's one-float buffers and its own stream (host path only).
    scratch: cuda.DeviceBuffer,
    stream: cuda.Stream,

    /// Joins the communicator `id` names (every rank calls it with the same id; it returns once all have).
    pub fn init(lib: *const nccl.Library, d: *const cuda.Driver, id: nccl.UniqueId, rank: u32, world: u32) !Nccl {
        var comm: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        errdefer _ = lib.api.ncclCommDestroy(comm);
        var scratch = try cuda.DeviceBuffer.alloc(d, 4 * (world + 1));
        errdefer scratch.free();
        try scratch.fill8(0, null);
        return .{ .lib = lib, .d = d, .comm = comm, .me = rank, .size = world, .scratch = scratch, .stream = try cuda.Stream.init(d, true) };
    }

    pub fn deinit(n: *Nccl) void {
        _ = n.stream.synchronize() catch {};
        n.stream.deinit();
        n.scratch.free();
        if (!n.aborted.load(.acquire)) _ = n.lib.api.ncclCommDestroy(n.comm);
    }

    pub fn iface(n: *Nccl) collective.Collective {
        return .{ .ptr = n, .vtable = &vtable };
    }

    const vtable: collective.Collective.VTable = .{
        .kind = .nccl,
        .rank = rankOf,
        .world = worldOf,
        .all_reduce = allReduce,
        .all_gather = allGather,
        .exchange = exchange,
        .send = send,
        .recv = recv,
        .barrier = barrier,
        .check = check,
        .abort = abort,
    };

    fn self_(ptr: *anyopaque) *Nccl {
        return @ptrCast(@alignCast(ptr));
    }

    fn rankOf(ptr: *anyopaque) u32 {
        return self_(ptr).me;
    }

    fn worldOf(ptr: *anyopaque) u32 {
        return self_(ptr).size;
    }

    fn call(n: *Nccl, r: nccl.Result, what: []const u8) Error!void {
        if (r == 0) return;
        if (n.aborted.load(.acquire)) return error.Aborted;
        n.lib.check(r, what) catch return error.BackendFailed;
    }

    fn live(n: *Nccl) Error!void {
        if (n.aborted.load(.acquire)) return error.Aborted;
    }

    pub fn allReduce(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, op: Op, stream: collective.Stream) Error!void {
        const n = self_(ptr);
        try n.live();
        try n.call(n.lib.api.ncclAllReduce(s, r, count, dataType(t), redOp(op), n.comm, stream), "ncclAllReduce");
    }

    pub fn allGather(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, stream: collective.Stream) Error!void {
        const n = self_(ptr);
        try n.live();
        try n.call(n.lib.api.ncclAllGather(s, r, count, dataType(t), n.comm, stream), "ncclAllGather");
    }

    pub fn exchange(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, peer: u32, stream: collective.Stream) Error!void {
        const n = self_(ptr);
        try n.live();
        try collective.checkPeer(n.me, n.size, peer);
        try n.call(n.lib.api.ncclGroupStart(), "ncclGroupStart");
        const a = n.lib.api.ncclSend(s, count, dataType(t), @intCast(peer), n.comm, stream);
        const b = n.lib.api.ncclRecv(r, count, dataType(t), @intCast(peer), n.comm, stream);
        const e = n.lib.api.ncclGroupEnd();
        try n.call(a, "ncclSend");
        try n.call(b, "ncclRecv");
        try n.call(e, "ncclGroupEnd");
    }

    fn send(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: collective.Stream) Error!void {
        const n = self_(ptr);
        try n.live();
        try collective.checkPeer(n.me, n.size, peer);
        try n.call(n.lib.api.ncclSend(buf, count, dataType(t), @intCast(peer), n.comm, stream), "ncclSend");
    }

    fn recv(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: collective.Stream) Error!void {
        const n = self_(ptr);
        try n.live();
        try collective.checkPeer(n.me, n.size, peer);
        try n.call(n.lib.api.ncclRecv(buf, count, dataType(t), @intCast(peer), n.comm, stream), "ncclRecv");
    }

    /// One float all-gathered on the barrier's own stream, then a sync of it (host path, never captured).
    fn barrier(ptr: *anyopaque) Error!void {
        const n = self_(ptr);
        try n.live();
        try n.call(n.lib.api.ncclAllGather(n.scratch.ptr, n.scratch.ptr + 4, 1, .f32, n.comm, n.stream.handle), "ncclAllGather");
        n.stream.synchronize() catch return if (n.aborted.load(.acquire)) error.Aborted else error.BackendFailed;
    }

    fn check(ptr: *anyopaque) Error!void {
        const n = self_(ptr);
        try n.live();
        var r: nccl.Result = 0;
        try n.call(n.lib.api.ncclCommGetAsyncError(n.comm, &r), "ncclCommGetAsyncError");
        if (r != 0 and r != nccl.in_progress) {
            n.lib.check(r, "NCCL async error") catch {};
            return error.PeerFailed;
        }
    }

    fn abort(ptr: *anyopaque) void {
        const n = self_(ptr);
        if (n.aborted.swap(true, .acq_rel)) return;
        _ = n.lib.api.ncclCommAbort(n.comm);
    }
};
