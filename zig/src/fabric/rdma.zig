//! One-sided direct memory access between ranks' registered windows: the interface every link implements.
const std = @import("std");

pub const page = std.heap.page_size_min;

/// Which link a rank pair uses; the cost model prices each.
pub const LinkKind = enum {
    /// Same node: a copy within unified memory.
    local,
    /// Mac to Mac over Thunderbolt 5: MCDMA emulates ordered window writes with two-sided receives and progress.
    tb5,
    /// Mac to a CUDA host over MCDMA's ConnectX link.
    cx5,
    /// CUDA to CUDA through NCCL on the hosts' ConnectX-7.
    nccl,
};

/// What a window may be used for by peers, as an MR's access flags.
pub const Access = packed struct(u8) { remote_write: bool = true, remote_read: bool = true, remote_atomic: bool = true, _: u5 = 0 };

pub const Error = error{ OutOfBounds, AccessDenied, Unaligned, PeerDown, NoSuchRank, Timeout, ClockUnavailable };

/// One rank's endpoint. Writes to one peer land in posting order (RC); writes to different peers are unordered.
pub const Rdma = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        rank: *const fn (ptr: *anyopaque) u32,
        size: *const fn (ptr: *anyopaque) u32,
        /// This rank's registered window; every rank's window has the same layout and length.
        window: *const fn (ptr: *anyopaque) []align(page) u8,
        /// Copy `bytes` into the peer's window at `offset`.
        write: *const fn (ptr: *anyopaque, peer: u32, offset: usize, bytes: []const u8) Error!void,
        /// `head` then `body` as one contiguous write when the link can (one message instead of two).
        write2: ?*const fn (ptr: *anyopaque, peer: u32, offset: usize, head: []const u8, body: []const u8) Error!void = null,
        /// Store an aligned 64-bit word at the peer, after every earlier write to that peer.
        signal: *const fn (ptr: *anyopaque, peer: u32, offset: usize, value: u64) Error!void,
        /// `write2` then `signal` as one message when the link can.
        write2_signal: ?*const fn (ptr: *anyopaque, peer: u32, offset: usize, head: []const u8, body: []const u8, flag: usize, value: u64) Error!void = null,
        /// Atomic fetch-and-add on an aligned 64-bit word at the peer; returns the old value.
        fetch_add: *const fn (ptr: *anyopaque, peer: u32, offset: usize, add: u64) Error!u64,
        /// Copy the peer's window at `offset` into `out`.
        read: *const fn (ptr: *anyopaque, peer: u32, offset: usize, out: []u8) Error!void,
        /// Return once every posted operation has completed at its target.
        flush: *const fn (ptr: *anyopaque) Error!void,
        link: *const fn (ptr: *anyopaque, peer: u32) LinkKind,
        deadline: ?*const fn (ptr: *anyopaque, deadline_ns: u64) void = null,
    };

    pub fn rank(r: Rdma) u32 {
        return r.vtable.rank(r.ptr);
    }
    pub fn size(r: Rdma) u32 {
        return r.vtable.size(r.ptr);
    }
    pub fn window(r: Rdma) []align(page) u8 {
        return r.vtable.window(r.ptr);
    }
    pub fn write(r: Rdma, peer: u32, offset: usize, bytes: []const u8) Error!void {
        return r.vtable.write(r.ptr, peer, offset, bytes);
    }
    pub fn write2(r: Rdma, peer: u32, offset: usize, head: []const u8, body: []const u8) Error!void {
        if (r.vtable.write2) |both| return both(r.ptr, peer, offset, head, body);
        try r.write(peer, offset, head);
        if (body.len > 0) try r.write(peer, offset + head.len, body);
    }
    pub fn signal(r: Rdma, peer: u32, offset: usize, value: u64) Error!void {
        return r.vtable.signal(r.ptr, peer, offset, value);
    }
    pub fn write2Signal(r: Rdma, peer: u32, offset: usize, head: []const u8, body: []const u8, flag: usize, value: u64) Error!void {
        if (r.vtable.write2_signal) |both| return both(r.ptr, peer, offset, head, body, flag, value);
        try r.write2(peer, offset, head, body);
        try r.signal(peer, flag, value);
    }
    pub fn fetchAdd(r: Rdma, peer: u32, offset: usize, add: u64) Error!u64 {
        return r.vtable.fetch_add(r.ptr, peer, offset, add);
    }
    pub fn read(r: Rdma, peer: u32, offset: usize, out: []u8) Error!void {
        return r.vtable.read(r.ptr, peer, offset, out);
    }
    pub fn flush(r: Rdma) Error!void {
        return r.vtable.flush(r.ptr);
    }
    pub fn link(r: Rdma, peer: u32) LinkKind {
        return r.vtable.link(r.ptr, peer);
    }

    pub fn setDeadline(r: Rdma, deadline_ns: u64) void {
        if (r.vtable.deadline) |set| set(r.ptr, deadline_ns);
    }

    /// An acquire load of a word in this rank's own window (where peers' signals land).
    pub fn local(r: Rdma, offset: usize) u64 {
        const w: *const u64 = @ptrCast(@alignCast(r.window().ptr + offset));
        return @atomicLoad(u64, w, .acquire);
    }
};
