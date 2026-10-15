//! What a tensor-parallel engine asks of its ranks: collectives on its streams, and rank 0's words to the others.

const std = @import("std");
const abi = @import("abi.zig");

const agreement = @import("agree.zig");

pub const Error = error{CollectiveFailed};
pub const LinkError = error{LinkFailed} || std.mem.Allocator.Error;
/// Why a rank could not join its group: no tensor parallelism in this build, no RCCL, the link, or a rank that differs.
pub const JoinError = error{
    NoTensorParallel,
    NoRccl,
    NoId,
    SocketFailed,
    BindFailed,
    ConnectFailed,
    LinkClosed,
    BadRank,
    RanksDiffer,
} || std.mem.Allocator.Error;

pub const Identity = agreement.Identity;
pub const Field = agreement.Field;
pub const Fault = agreement.Fault;
pub const Step = agreement.Step;
pub const Verdict = agreement.Verdict;
pub const How = agreement.How;
pub const Vote = agreement.Vote;
pub const tally = agreement.tally;

/// What a rank brings to its join: its place, its GPU and policy, what it runs, and where a differing rank is named.
pub const Join = struct {
    rank: u32,
    world: u32,
    master: []const u8,
    port: u16,
    caps: @import("caps.zig").Caps,
    policy: *@import("policy.zig").Policy,
    ident: Identity,
    fault: *Fault,
};

/// The join of a build without a collectives backend: one rank only.
pub fn noGroup(_: std.mem.Allocator, _: std.Io, _: Join) JoinError!Group {
    return error.NoTensorParallel;
}

/// The element types the forward moves, in RCCL's numbering.
pub const Dtype = enum(c_int) { u8 = 1, i32 = 2, i64 = 4, f16 = 6, f32 = 7, bf16 = 9 };

/// One rank's communicator. A collective queues on its stream and waits for every rank's matching call.
pub const Comm = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    rank: usize,
    world: usize,

    pub const VTable = struct {
        all_reduce: *const fn (ptr: *anyopaque, send: u64, recv: u64, n: usize, dtype: Dtype, stream: abi.Stream) Error!void,
        all_gather: *const fn (ptr: *anyopaque, send: u64, recv: u64, n: usize, dtype: Dtype, stream: abi.Stream) Error!void,
        broadcast: *const fn (ptr: *anyopaque, send: u64, recv: u64, n: usize, dtype: Dtype, root: usize, stream: abi.Stream) Error!void,
        close: *const fn (ptr: *anyopaque) void,
    };

    /// `recv` = the ranks' `send` summed, `n` values of `dtype`.
    pub fn allReduce(c: Comm, send: u64, recv: u64, n: usize, dtype: Dtype, stream: abi.Stream) Error!void {
        return c.vtable.all_reduce(c.ptr, send, recv, n, dtype, stream);
    }

    /// `recv` (world * n values) = every rank's `send` (n values) in rank order.
    pub fn allGather(c: Comm, send: u64, recv: u64, n: usize, dtype: Dtype, stream: abi.Stream) Error!void {
        return c.vtable.all_gather(c.ptr, send, recv, n, dtype, stream);
    }

    /// `root`'s `send` into every rank's `recv`, `n` values.
    pub fn broadcast(c: Comm, send: u64, recv: u64, n: usize, dtype: Dtype, root: usize, stream: abi.Stream) Error!void {
        return c.vtable.broadcast(c.ptr, send, recv, n, dtype, root, stream);
    }

    pub fn close(c: Comm) void {
        c.vtable.close(c.ptr);
    }
};

/// Rank 0's channel to the others: it sends each step, a follower takes them in order.
pub const Link = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        send: *const fn (ptr: *anyopaque, words: []const u32) LinkError!void,
        recv: *const fn (ptr: *anyopaque, gpa: std.mem.Allocator, out: *std.ArrayList(u32)) LinkError!void,
        agree: *const fn (ptr: *anyopaque, mine: Step, how: How) LinkError!Verdict,
    };

    /// Rank 0: `words` to every follower.
    pub fn send(l: Link, words: []const u32) LinkError!void {
        return l.vtable.send(l.ptr, words);
    }

    /// A follower: rank 0's next message into `out` (cleared first).
    pub fn recv(l: Link, gpa: std.mem.Allocator, out: *std.ArrayList(u32)) LinkError!void {
        return l.vtable.recv(l.ptr, gpa, out);
    }

    /// A startup step every rank takes: this rank's outcome in, every rank's verdict out (a closed link refuses).
    pub fn agree(l: Link, mine: Step, how: How) LinkError!Verdict {
        return l.vtable.agree(l.ptr, mine, how);
    }
};

/// What a rank holds once it joined its group: the link, and its communicator opened when the engine's GPU is current.
pub const Group = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    rank: usize,
    world: usize,

    pub const VTable = struct {
        link: *const fn (ptr: *anyopaque) Link,
        open: *const fn (ptr: *anyopaque) Error!Comm,
        close: *const fn (ptr: *anyopaque, gpa: std.mem.Allocator) void,
    };

    pub fn link(g: Group) Link {
        return g.vtable.link(g.ptr);
    }

    /// This rank's communicator: blocks until every rank has opened its own; the engine's device is current.
    pub fn open(g: Group) Error!Comm {
        return g.vtable.open(g.ptr);
    }

    pub fn close(g: Group, gpa: std.mem.Allocator) void {
        g.vtable.close(g.ptr, gpa);
    }
};

test "a comm forwards each collective to its implementation" {
    const Fake = struct {
        calls: [3]usize = .{ 0, 0, 0 },
        closed: bool = false,
        fn reduce(ptr: *anyopaque, _: u64, _: u64, _: usize, _: Dtype, _: abi.Stream) Error!void {
            @as(*@This(), @ptrCast(@alignCast(ptr))).calls[0] += 1;
        }
        fn gather(ptr: *anyopaque, _: u64, _: u64, _: usize, _: Dtype, _: abi.Stream) Error!void {
            @as(*@This(), @ptrCast(@alignCast(ptr))).calls[1] += 1;
        }
        fn cast(ptr: *anyopaque, _: u64, _: u64, _: usize, _: Dtype, _: usize, _: abi.Stream) Error!void {
            @as(*@This(), @ptrCast(@alignCast(ptr))).calls[2] += 1;
        }
        fn shut(ptr: *anyopaque) void {
            @as(*@This(), @ptrCast(@alignCast(ptr))).closed = true;
        }
    };
    var f: Fake = .{};
    const c: Comm = .{ .ptr = &f, .vtable = &.{ .all_reduce = Fake.reduce, .all_gather = Fake.gather, .broadcast = Fake.cast, .close = Fake.shut }, .rank = 1, .world = 2 };
    try c.allReduce(0, 0, 1, .f32, null);
    try c.allGather(0, 0, 1, .bf16, null);
    try c.allGather(0, 0, 1, .bf16, null);
    try c.broadcast(0, 0, 1, .i32, 0, null);
    c.close();
    try std.testing.expectEqual([3]usize{ 1, 2, 1 }, f.calls);
    try std.testing.expect(f.closed);
}
