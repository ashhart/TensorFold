//! The production warm-up posts through fake NCCL: topology, offsets, selectors and failure closure.
const std = @import("std");
const cuda = @import("cuda");
const warm = @import("cuda_warm_peers.zig");
const ranks = @import("cuda_ranks.zig");
const t = std.testing;

const Fake = struct {
    const Kind = enum { start, end, send, recv };
    const Call = struct { kind: Kind, peer: usize = 0, pointer: u64 = 0, bytes: usize = 0 };
    var calls: [64]Call = undefined;
    var count: usize = 0;
    var fail: ?usize = null;
    var end_fails = false;
    var mode: ?[]const u8 = null;

    fn note(c: Call) cuda.nccl.Result {
        calls[count] = c;
        count += 1;
        return if (fail == count - 1 or (end_fails and c.kind == .end)) 1 else 0;
    }
    fn start() callconv(.c) cuda.nccl.Result {
        return note(.{ .kind = .start });
    }
    fn end() callconv(.c) cuda.nccl.Result {
        return note(.{ .kind = .end });
    }
    fn send(ptr: u64, n: usize, _: cuda.nccl.DataType, peer: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        return note(.{ .kind = .send, .peer = @intCast(peer), .pointer = ptr, .bytes = n });
    }
    fn recv(ptr: u64, n: usize, _: cuda.nccl.DataType, peer: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        return note(.{ .kind = .recv, .peer = @intCast(peer), .pointer = ptr, .bytes = n });
    }
    fn group(rank: usize, world: usize) cuda.tp_group.Group {
        var api: cuda.nccl.Api = undefined;
        api.ncclGroupStart = start;
        api.ncclGroupEnd = end;
        api.ncclSend = send;
        api.ncclRecv = recv;
        return .{ .rank = rank, .world = world, .comm = null, .lib = .{ .lib = undefined, .api = api } };
    }
    fn reset() void {
        count = 0;
        fail = null;
        end_fails = false;
    }
    fn get(name: [:0]const u8) ?[]const u8 {
        return if (std.mem.eql(u8, name, "TF_GLM_PREFILL_REDUCE")) mode else null;
    }
    fn run(g: *const cuda.tp_group.Group, ring: bool) !void {
        try warm.post(g, .{ .d = undefined, .handle = null }, 0x4000, ring);
    }
};

test "GLM warm peers: ring has only neighbours, direct has every peer, all postings match and buffers stay distinct" {
    for (2..9) |world| for ([_]bool{ false, true }) |ring| {
        var sends: [8][8]u8 = @splat(@splat(0));
        var recvs: [8][8]u8 = @splat(@splat(0));
        for (0..world) |rank| {
            const g = Fake.group(rank, world);
            Fake.reset();
            try Fake.run(&g, ring);
            const peers = if (ring) @min(world - 1, 2) else world - 1;
            try t.expectEqual(2 + 2 * peers, Fake.count);
            try t.expectEqual(Fake.Kind.start, Fake.calls[0].kind);
            try t.expectEqual(Fake.Kind.end, Fake.calls[Fake.count - 1].kind);
            for (Fake.calls[1 .. Fake.count - 1]) |c| {
                try t.expect(c.peer != rank and c.peer < world);
                if (ring) try t.expect(c.peer == (rank + 1) % world or c.peer == (rank + world - 1) % world);
                try t.expectEqual(@as(usize, 4 << 20), c.bytes);
                switch (c.kind) {
                    .send => {
                        sends[rank][c.peer] += 1;
                        try t.expectEqual(@as(u64, 0x4000 + c.peer * warm.bytes), c.pointer);
                    },
                    .recv => {
                        recvs[rank][c.peer] += 1;
                        try t.expectEqual(@as(u64, 0x4000 + (world + c.peer) * warm.bytes), c.pointer);
                    },
                    else => return error.UnexpectedGroup,
                }
            }
        }
        for (0..world) |rank| for (0..world) |peer| {
            try t.expect(sends[rank][peer] <= 1);
            try t.expectEqual(sends[rank][peer], recvs[peer][rank]);
            const wanted = peer != rank and (!ring or peer == (rank + 1) % world or peer == (rank + world - 1) % world);
            try t.expectEqual(@as(u8, @intFromBool(wanted)), sends[rank][peer]);
        };
    };
}

test "GLM warm peers: failure at every posting closes only an opened group and sends nothing after the failure" {
    for (2..9) |world| for (0..world) |rank| for ([_]bool{ false, true }) |ring| {
        const g = Fake.group(rank, world);
        Fake.reset();
        try Fake.run(&g, ring);
        const total = Fake.count;
        const clean = Fake.calls;
        for (0..total) |fail| {
            Fake.reset();
            Fake.fail = fail;
            try t.expectError(error.NcclFailed, Fake.run(&g, ring));
            const posting = clean[fail].kind == .send or clean[fail].kind == .recv;
            try t.expectEqual(fail + 1 + @intFromBool(posting), Fake.count);
            for (Fake.calls[0 .. fail + 1], clean[0 .. fail + 1]) |got, wanted| try t.expectEqualDeep(wanted, got);
            if (posting) try t.expectEqual(Fake.Kind.end, Fake.calls[Fake.count - 1].kind);
        }
        Fake.reset();
        Fake.fail = 1;
        Fake.end_fails = true;
        try t.expectError(error.NcclFailed, Fake.run(&g, ring));
        try t.expectEqual(@as(usize, 3), Fake.count);
    };
}

test "GLM warm peers: one rank does nothing and malformed placement refuses before NCCL" {
    Fake.reset();
    var g = Fake.group(0, 1);
    try Fake.run(&g, true);
    try Fake.run(&g, false);
    try t.expectEqual(@as(usize, 0), Fake.count);
    g.world = 0;
    try t.expectError(error.BadWorld, Fake.run(&g, true));
    g.world = 2;
    g.rank = 2;
    try t.expectError(error.BadWorld, Fake.run(&g, false));
    try t.expectEqual(@as(usize, 0), Fake.count);
}

test "GLM warm peers: existing ring selector alone restricts warm-up, gather and default split keep every peer" {
    for ([_]?[]const u8{ null, "split", "gather", "ring" }) |mode| {
        Fake.mode = mode;
        var why: []const u8 = "";
        const settings = try ranks.Settings.read(Fake.get, &why);
        const ring = if (mode) |v| std.mem.eql(u8, v, "ring") else false;
        try t.expectEqual(ring, settings.ring);
        const g = Fake.group(0, 4);
        Fake.reset();
        try Fake.run(&g, settings.ring);
        try t.expectEqual(@as(usize, if (ring) 6 else 8), Fake.count);
    }
    Fake.mode = null;
}
