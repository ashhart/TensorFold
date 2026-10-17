//! A tensor-parallel group over NCCL: NCCL_COMM_ID gives every rank the same id; rows meet in rank order.
const std = @import("std");
const nccl = @import("nccl.zig");
const driver = @import("driver.zig");
const stream_ = @import("stream.zig");
const memory = @import("memory.zig");

const Stream = stream_.Stream;

pub const Group = struct {
    lib: nccl.Library,
    comm: nccl.Comm,
    rank: usize,
    world: usize,

    /// Rank `rank` of `world` (NCCL opened at run time; NCCL_COMM_ID names rank 0's address, every rank the same).
    pub fn open(rank: usize, world: usize) !Group {
        if (std.c.getenv("NCCL_COMM_ID") == null) return error.NoCommId;
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        var comm: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = comm, .rank = rank, .world = world };
    }

    pub fn close(g: *Group) void {
        _ = g.lib.api.ncclCommDestroy(g.comm);
        g.lib.close();
        g.* = undefined;
    }

    /// `bytes` at `mine` from every rank into `all` in rank order (world * bytes there), on `s`.
    pub fn allGather(g: *const Group, s: Stream, mine: u64, all: u64, bytes: usize) !void {
        try g.lib.check(g.lib.api.ncclAllGather(mine, all, bytes, .u8, g.comm, s.handle), "ncclAllGather");
    }

    /// `bytes` at `ptr` to rank `peer`, on `s` (its matching recv takes the same count).
    pub fn send(g: *const Group, s: Stream, ptr: u64, bytes: usize, peer: usize) !void {
        try g.lib.check(g.lib.api.ncclSend(ptr, bytes, .u8, @intCast(peer), g.comm, s.handle), "ncclSend");
    }

    /// `bytes` from rank `peer` into `ptr`, on `s`.
    pub fn recv(g: *const Group, s: Stream, ptr: u64, bytes: usize, peer: usize) !void {
        try g.lib.check(g.lib.api.ncclRecv(ptr, bytes, .u8, @intCast(peer), g.comm, s.handle), "ncclRecv");
    }

    /// Rank 0 and rank `peer` connect both ways with one small exchange (`ptr`: 128 bytes, 64 out and 64 in).
    pub fn connect(g: *const Group, s: Stream, ptr: u64, peer: usize) !void {
        try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
        const posted: anyerror!void = post: {
            g.send(s, ptr, 64, peer) catch |err| break :post err;
            g.recv(s, ptr + 64, 64, peer) catch |err| break :post err;
            break :post {};
        };
        try g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd"); // the group closes on every path
        try posted;
        try s.synchronize();
    }

    /// Every rank's `mine` in rank order into `out` (world of them), waited for.
    pub fn gather(g: *const Group, comptime T: type, d: *const driver.Driver, s: Stream, mine: T, out: []T) !void {
        if (out.len != g.world) return error.BadWorld;
        const size = @sizeOf(T);
        var dev = try memory.DeviceBuffer.alloc(d, size * g.world);
        defer dev.free();
        try dev.uploadAsync(g.rank * size, std.mem.asBytes(&mine), s.handle);
        try g.allGather(s, dev.ptr + g.rank * size, dev.ptr, size);
        try s.synchronize();
        try dev.download(0, std.mem.sliceAsBytes(out));
    }
};
