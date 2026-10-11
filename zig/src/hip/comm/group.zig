//! A tensor-parallel group over RCCL: the TCP link that carries rank 0's steps and RCCL's id, behind collective.zig.

const std = @import("std");
const collective = @import("../collective.zig");
const rccl_mod = @import("rccl.zig");
const link_mod = @import("link.zig");
const Caps = @import("../caps.zig").Caps;
const Policy = @import("../policy.zig").Policy;
const Device = @import("../device.zig").Device;

/// RCCL's library stays loaded until the group is closed (its bootstrap thread starts at the id).
const Joined = struct {
    rccl: rccl_mod.Rccl,
    link: link_mod.Link,
    id: rccl_mod.UniqueId,
    rank: usize,
    world: usize,
    comm: ?rccl_mod.Comm = null,
};

/// Rank 0 listens on `master`:`port`, sends its id and policy; a rank whose GPU differs from rank 0's is refused.
pub fn join(gpa: std.mem.Allocator, io: std.Io, rank: u32, world: u32, master: []const u8, port: u16, caps: Caps, policy: *Policy) collective.JoinError!collective.Group {
    const g = try gpa.create(Joined);
    errdefer gpa.destroy(g);
    g.rccl = rccl_mod.Rccl.open(policy.rccl_lib.slice()) catch return error.NoRccl;
    errdefer g.rccl.close();
    const mine: rccl_mod.UniqueId = if (rank == 0) g.rccl.uniqueId() catch return error.NoId else undefined;
    const host = if (std.mem.eql(u8, master, "localhost")) "127.0.0.1" else master;
    const pair = try link_mod.Link.open(io, rank, world, host, port, mine, .{ .caps = Device.groupId(caps), .policy = policy.words() });
    g.link, g.id, const hello = pair;
    g.rank = rank;
    g.world = world;
    g.comm = null;
    if (rank != 0) policy.* = Policy.fromWords(hello.policy, policy.*);
    return .{ .ptr = g, .vtable = &group_vtable, .rank = rank, .world = world };
}

fn of(ptr: *anyopaque) *Joined {
    return @ptrCast(@alignCast(ptr));
}

const group_vtable: collective.Group.VTable = .{ .link = linkOf, .open = open, .close = close };
const link_vtable: collective.Link.VTable = .{ .send = send, .recv = recv };
const comm_vtable: collective.Comm.VTable = .{ .all_reduce = allReduce, .all_gather = allGather, .broadcast = broadcast, .close = closeComm };

fn linkOf(ptr: *anyopaque) collective.Link {
    return .{ .ptr = &of(ptr).link, .vtable = &link_vtable };
}

/// The rank's communicator, opened once: ncclCommInitRank blocks until every rank has called it.
fn open(ptr: *anyopaque) collective.Error!collective.Comm {
    const g = of(ptr);
    if (g.comm == null) g.comm = rccl_mod.Comm.init(&g.rccl, g.id, g.rank, g.world) catch return error.CollectiveFailed;
    return .{ .ptr = &g.comm.?, .vtable = &comm_vtable, .rank = g.rank, .world = g.world };
}

fn close(ptr: *anyopaque, gpa: std.mem.Allocator) void {
    const g = of(ptr);
    if (g.comm) |*c| c.deinit();
    g.link.close();
    g.rccl.close();
    gpa.destroy(g);
}

fn send(ptr: *anyopaque, words: []const u32) collective.LinkError!void {
    const l: *const link_mod.Link = @ptrCast(@alignCast(ptr));
    l.send(words) catch return error.LinkFailed;
}

fn recv(ptr: *anyopaque, gpa: std.mem.Allocator, out: *std.ArrayList(u32)) collective.LinkError!void {
    const l: *const link_mod.Link = @ptrCast(@alignCast(ptr));
    l.recv(gpa, out) catch |err| return if (err == error.OutOfMemory) error.OutOfMemory else error.LinkFailed;
}

fn comm(ptr: *anyopaque) *rccl_mod.Comm {
    return @ptrCast(@alignCast(ptr));
}

fn allReduce(ptr: *anyopaque, s: u64, r: u64, n: usize, dtype: collective.Dtype, stream: @import("../abi.zig").Stream) collective.Error!void {
    comm(ptr).allReduce(s, r, n, dtype, stream) catch return error.CollectiveFailed;
}

fn allGather(ptr: *anyopaque, s: u64, r: u64, n: usize, dtype: collective.Dtype, stream: @import("../abi.zig").Stream) collective.Error!void {
    comm(ptr).allGather(s, r, n, dtype, stream) catch return error.CollectiveFailed;
}

fn broadcast(ptr: *anyopaque, s: u64, r: u64, n: usize, dtype: collective.Dtype, root: usize, stream: @import("../abi.zig").Stream) collective.Error!void {
    comm(ptr).broadcast(s, r, n, dtype, root, stream) catch return error.CollectiveFailed;
}

/// The communicator lives with the group; the group's close destroys it.
fn closeComm(_: *anyopaque) void {}
