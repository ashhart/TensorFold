//! The ranks of a tensor-parallel group: the TCP link that carries rank 0's steps and RCCL's unique id.

const std = @import("std");
const rccl_mod = @import("rccl.zig");
const link_mod = @import("link.zig");
const Caps = @import("../caps.zig").Caps;
const Policy = @import("../policy.zig").Policy;

pub const Error = link_mod.Error || error{ NoRccl, NoId };

/// RCCL's library stays loaded until the group is closed (its bootstrap thread starts at the id).
pub const Group = struct {
    rccl: rccl_mod.Rccl,
    link: link_mod.Link,
    id: rccl_mod.UniqueId,

    /// Rank 0 listens on `master`:`port`, sends its id and policy; a rank whose GPU differs from rank 0's is refused.
    pub fn join(gpa: std.mem.Allocator, io: std.Io, rank: u32, world: u32, master: []const u8, port: u16, caps: Caps, policy: *Policy) Error!*Group {
        const g = try gpa.create(Group);
        errdefer gpa.destroy(g);
        g.rccl = rccl_mod.Rccl.open(policy.rccl_lib.slice()) catch return error.NoRccl;
        errdefer g.rccl.close();
        const mine: rccl_mod.UniqueId = if (rank == 0) g.rccl.uniqueId() catch return error.NoId else undefined;
        const host = if (std.mem.eql(u8, master, "localhost")) "127.0.0.1" else master;
        const pair = try link_mod.Link.open(io, rank, world, host, port, mine, .{ .caps = @import("../device.zig").Device.groupId(caps), .policy = policy.words() });
        g.link, g.id, const hello = pair;
        if (rank != 0) policy.* = Policy.fromWords(hello.policy, policy.*);
        return g;
    }

    pub fn close(g: *Group, gpa: std.mem.Allocator) void {
        g.link.close();
        g.rccl.close();
        gpa.destroy(g);
    }
};
