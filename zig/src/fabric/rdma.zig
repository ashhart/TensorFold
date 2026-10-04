//! One-sided direct memory access between ranks' registered windows: the interface every link implements.
const std = @import("std");

pub const page = std.heap.page_size_min;

/// Which link a rank pair uses; the cost model prices each.
pub const LinkKind = enum {
    /// Same node: a copy within unified memory.
    local,
    /// Mac to Mac over Thunderbolt 5 (Apple's rdma_en devices): two-sided UC only, so sendrecv.zig, not this interface.
    tb5,
    /// Mac to a CUDA host over MCDMA's ConnectX link.
    cx5,
    /// CUDA to CUDA through NCCL on the hosts' ConnectX-7.
    nccl,
};

/// What a window may be used for by peers, as an MR's access flags.
pub const Access = packed struct(u8) { remote_write: bool = true, remote_read: bool = true, remote_atomic: bool = true, _: u5 = 0 };

pub const Error = error{ OutOfBounds, AccessDenied, Unaligned, PeerDown, NoSuchRank };

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
        /// Store an aligned 64-bit word at the peer, after every earlier write to that peer.
        signal: *const fn (ptr: *anyopaque, peer: u32, offset: usize, value: u64) Error!void,
        /// Atomic fetch-and-add on an aligned 64-bit word at the peer; returns the old value.
        fetch_add: *const fn (ptr: *anyopaque, peer: u32, offset: usize, add: u64) Error!u64,
        /// Copy the peer's window at `offset` into `out`.
        read: *const fn (ptr: *anyopaque, peer: u32, offset: usize, out: []u8) Error!void,
        /// Return once every posted operation has completed at its target.
        flush: *const fn (ptr: *anyopaque) Error!void,
        link: *const fn (ptr: *anyopaque, peer: u32) LinkKind,
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
    pub fn signal(r: Rdma, peer: u32, offset: usize, value: u64) Error!void {
        return r.vtable.signal(r.ptr, peer, offset, value);
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

    /// An acquire load of a word in this rank's own window (where peers' signals land).
    pub fn local(r: Rdma, offset: usize) u64 {
        const w: *const u64 = @ptrCast(@alignCast(r.window().ptr + offset));
        return @atomicLoad(u64, w, .acquire);
    }
};
