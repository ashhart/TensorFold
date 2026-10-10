//! Small exchanges through the mailbox (one host) or RoCE (two hosts), the rest through NCCL; ranks always agree.
const std = @import("std");
const collective = @import("collective.zig");
const Mailbox = @import("mailbox.zig").Mailbox;
const Roce = @import("roce.zig").Roce;
const Nccl = @import("nccl.zig").Nccl;
const DevicePtr = collective.DevicePtr;
const DType = collective.DType;
const Op = collective.Op;
const Error = collective.Error;

/// The one-kernel path for small shards: both expose the same calls.
pub const Small = union(enum) {
    mailbox: *Mailbox,
    roce: *Roce,

    fn fits(s: Small, n: usize) bool {
        return switch (s) {
            inline else => |p| p.fits(n),
        };
    }

    fn sum(s: Small, src: DevicePtr, dst: DevicePtr, count: usize, t: DType, stream: collective.Stream) Error!void {
        return switch (s) {
            inline else => |p| p.sum(src, dst, count, t, stream),
        };
    }

    fn gather(s: Small, src: DevicePtr, dst: DevicePtr, n: usize, stream: collective.Stream) Error!void {
        return switch (s) {
            inline else => |p| p.allGatherBytes(src, dst, n, stream),
        };
    }

    fn exchange(s: Small, src: DevicePtr, dst: DevicePtr, n: usize, stream: collective.Stream) Error!void {
        return switch (s) {
            inline else => |p| p.exchangeBytes(src, dst, n, stream),
        };
    }

    fn check(s: Small) Error!void {
        return switch (s) {
            inline else => |p| p.check(),
        };
    }

    fn abort(s: Small) void {
        switch (s) {
            inline else => |p| p.abort(),
        }
    }
};

pub const Hybrid = struct {
    small: Small,
    nccl: *Nccl,
    /// Largest shard a rank sends through the small path.
    max_bytes: usize,
    /// Every rank's `varFits(16)`, agreed at `Session.open` (the bootstrap's all-gather): `varAgreed`'s answer.
    var_agreed: bool = false,

    pub fn iface(h: *Hybrid) collective.Collective {
        return .{ .ptr = h, .vtable = if (h.small == .roce) &roce_vtable else &mailbox_vtable };
    }

    fn vtable(comptime kind: collective.Kind) collective.Collective.VTable {
        return .{
            .kind = kind,
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
            .all_gather_v = allGatherV,
            .var_fits = varFits,
            .var_agreed = varAgreed,
        };
    }

    const mailbox_vtable = vtable(.hybrid);
    const roce_vtable = vtable(.roce);

    fn self_(ptr: *anyopaque) *Hybrid {
        return @ptrCast(@alignCast(ptr));
    }

    fn routed(h: *const Hybrid, n: usize) bool {
        return n <= h.max_bytes and h.small.fits(n);
    }

    fn rankOf(ptr: *anyopaque) u32 {
        return self_(ptr).nccl.me;
    }

    fn worldOf(ptr: *anyopaque) u32 {
        return self_(ptr).nccl.size;
    }

    fn allReduce(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, op: Op, stream: collective.Stream) Error!void {
        const h = self_(ptr);
        if (op == .sum and (t == .f32 or t == .bf16) and h.routed(count * t.size())) return h.small.sum(s, r, count, t, stream);
        return Nccl.allReduce(h.nccl, s, r, count, t, op, stream);
    }

    fn allGather(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, stream: collective.Stream) Error!void {
        const h = self_(ptr);
        if (h.routed(count * t.size())) return h.small.gather(s, r, count * t.size(), stream);
        return Nccl.allGather(h.nccl, s, r, count, t, stream);
    }

    /// RoCE moves each rank's device length; the mailbox and NCCL move whole strides (the same bytes where lens says).
    fn allGatherV(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, stride: usize, lens: DevicePtr, t: DType, stream: collective.Stream) Error!void {
        const h = self_(ptr);
        const n = stride * t.size();
        if (varFits(ptr, n)) return h.small.roce.allGatherV(s, r, n, lens, stream);
        return allGather(ptr, s, r, stride, t, stream);
    }

    /// RoCE with its device side, for a stride the small path takes (the route is the size's alone: every rank alike).
    pub fn varFits(ptr: *anyopaque, n: usize) bool {
        const h = self_(ptr);
        return switch (h.small) {
            .roce => |p| n > 0 and h.routed(n) and p.d != null,
            .mailbox => false,
        };
    }

    fn varAgreed(ptr: *anyopaque) Error!bool {
        return self_(ptr).var_agreed;
    }

    fn exchange(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, peer: u32, stream: collective.Stream) Error!void {
        const h = self_(ptr);
        try collective.checkPeer(h.nccl.me, h.nccl.size, peer);
        if (h.routed(count * t.size())) return h.small.exchange(s, r, count * t.size(), stream);
        return Nccl.exchange(h.nccl, s, r, count, t, peer, stream);
    }

    fn send(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: collective.Stream) Error!void {
        return Nccl.iface(self_(ptr).nccl).send(buf, count, t, peer, stream);
    }

    fn recv(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, stream: collective.Stream) Error!void {
        return Nccl.iface(self_(ptr).nccl).recv(buf, count, t, peer, stream);
    }

    fn barrier(ptr: *anyopaque) Error!void {
        return Nccl.iface(self_(ptr).nccl).barrier();
    }

    fn check(ptr: *anyopaque) Error!void {
        const h = self_(ptr);
        try h.small.check();
        try Nccl.iface(h.nccl).check();
    }

    fn abort(ptr: *anyopaque) void {
        const h = self_(ptr);
        h.small.abort();
        Nccl.iface(h.nccl).abort();
    }
};
