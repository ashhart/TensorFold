//! The tensor-parallel communicator every backend implements: stream-ordered, graph-capturable, rank-order sums.
const std = @import("std");
const cuda = @import("cuda");

pub const DevicePtr = cuda.abi.DevicePtr;
pub const Stream = cuda.abi.Stream;

pub const Error = error{ Aborted, Unsupported, Invalid, PeerFailed, Timeout, BackendFailed };

/// Element types the collectives move (`reduce` takes the float ones and the 32/64-bit integers).
pub const DType = enum(u8) {
    u8,
    i32,
    i64,
    f16,
    bf16,
    f32,
    f64,

    pub fn size(t: DType) usize {
        return switch (t) {
            .u8 => 1,
            .f16, .bf16 => 2,
            .i32, .f32 => 4,
            .i64, .f64 => 8,
        };
    }
};

pub const Op = enum(u8) { sum, max, min };

/// Which implementation is behind a `Collective` (logs, bench labels).
pub const Kind = enum { nccl, mailbox, hybrid, roce, host };

pub const Collective = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        kind: Kind,
        rank: *const fn (ptr: *anyopaque) u32,
        world: *const fn (ptr: *anyopaque) u32,
        /// recv[count] <- the element-wise reduction of every rank's send[count], added in rank order.
        all_reduce: *const fn (ptr: *anyopaque, send: DevicePtr, recv: DevicePtr, count: usize, t: DType, op: Op, stream: Stream) Error!void,
        /// recv[world * count] <- every rank's send[count], rank 0's first (NCCL's layout).
        all_gather: *const fn (ptr: *anyopaque, send: DevicePtr, recv: DevicePtr, count: usize, t: DType, stream: Stream) Error!void,
        /// Two ranks trade: this rank's send[count] lands in `peer`'s recv[count] and the peer's in ours (one group).
        exchange: *const fn (ptr: *anyopaque, send: DevicePtr, recv: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void,
        send: *const fn (ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void,
        recv: *const fn (ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void,
        /// Host: returns once every rank has called it and this rank's earlier collectives have finished.
        barrier: *const fn (ptr: *anyopaque) Error!void,
        /// Host: a failure the transport recorded since the last check (a peer that timed out, an NCCL async error).
        check: *const fn (ptr: *anyopaque) Error!void,
        /// Any thread: end pending and later work at once; safe while another thread is inside a call.
        abort: *const fn (ptr: *anyopaque) void,
        /// recv[world][stride]: rank r's first lens[r] bytes (device int32 [world]) land at recv + r * stride.
        all_gather_v: ?*const fn (ptr: *anyopaque, send: DevicePtr, recv: DevicePtr, stride: usize, lens: DevicePtr, t: DType, stream: Stream) Error!void = null,
        /// Whether `all_gather_v` moves only the lengths for shards of `bytes` here (rank-local). Null: never.
        var_fits: ?*const fn (ptr: *anyopaque, bytes: usize) bool = null,
        /// Host: whether every rank's transport takes device lengths, agreed once per communicator. Null: false.
        var_agreed: ?*const fn (ptr: *anyopaque) Error!bool = null,
    };

    pub fn kind(c: Collective) Kind {
        return c.vtable.kind;
    }

    pub fn rank(c: Collective) u32 {
        return c.vtable.rank(c.ptr);
    }

    pub fn world(c: Collective) u32 {
        return c.vtable.world(c.ptr);
    }

    pub fn allReduce(c: Collective, src: DevicePtr, dst: DevicePtr, count: usize, t: DType, op: Op, stream: Stream) Error!void {
        return c.vtable.all_reduce(c.ptr, src, dst, count, t, op, stream);
    }

    pub fn allGather(c: Collective, src: DevicePtr, dst: DevicePtr, count: usize, t: DType, stream: Stream) Error!void {
        return c.vtable.all_gather(c.ptr, src, dst, count, t, stream);
    }

    /// `allGather` of `stride` elements a rank where only rank r's first lens[r] bytes matter (`VTable.all_gather_v`).
    pub fn allGatherV(c: Collective, src: DevicePtr, dst: DevicePtr, stride: usize, lens: DevicePtr, t: DType, stream: Stream) Error!void {
        if (c.vtable.all_gather_v) |f| return f(c.ptr, src, dst, stride, lens, t, stream);
        return c.vtable.all_gather(c.ptr, src, dst, stride, t, stream);
    }

    /// Whether `allGatherV` of `bytes`-byte strides moves only the lengths here (callers agree first: `varAgreed`).
    pub fn varFits(c: Collective, bytes: usize) bool {
        const f = c.vtable.var_fits orelse return false;
        return f(c.ptr, bytes);
    }

    /// Every rank's transport takes device lengths: one agreement a communicator, called alike outside any graph.
    pub fn varAgreed(c: Collective) Error!bool {
        const f = c.vtable.var_agreed orelse return false;
        return f(c.ptr);
    }

    pub fn exchange(c: Collective, src: DevicePtr, dst: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void {
        return c.vtable.exchange(c.ptr, src, dst, count, t, peer, stream);
    }

    pub fn send(c: Collective, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void {
        return c.vtable.send(c.ptr, buf, count, t, peer, stream);
    }

    pub fn recv(c: Collective, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: Stream) Error!void {
        return c.vtable.recv(c.ptr, buf, count, t, peer, stream);
    }

    pub fn barrier(c: Collective) Error!void {
        return c.vtable.barrier(c.ptr);
    }

    pub fn check(c: Collective) Error!void {
        return c.vtable.check(c.ptr);
    }

    pub fn abort(c: Collective) void {
        c.vtable.abort(c.ptr);
    }
};

/// Refuses a peer that is this rank or out of range.
pub fn checkPeer(rank: u32, world: u32, peer: u32) Error!void {
    if (peer == rank or peer >= world) return error.Invalid;
}

test "dtype sizes" {
    try std.testing.expectEqual(@as(usize, 2), DType.bf16.size());
    try std.testing.expectEqual(@as(usize, 8), DType.i64.size());
}
