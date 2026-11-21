//! The production warm-up posts through fake NCCL: topology, offsets, selectors and failure closure.
const std = @import("std");
const cuda = @import("cuda");
const warm = @import("cuda_warm_peers.zig");
const ranks = @import("cuda_ranks.zig");
const t = std.testing;

const Fake = struct {
    const Kind = enum { start, end, send, recv, gather };
    const Call = struct { kind: Kind, peer: usize = 0, pointer: u64 = 0, bytes: usize = 0, to: u64 = 0 };
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
    fn allGather(mine: u64, all: u64, n: usize, _: cuda.nccl.DataType, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        return note(.{ .kind = .gather, .pointer = mine, .bytes = n, .to = all });
    }
    fn group(rank: usize, world: usize) cuda.tp_group.Group {
        var api: cuda.nccl.Api = undefined;
        api.ncclGroupStart = start;
        api.ncclGroupEnd = end;
        api.ncclSend = send;
        api.ncclRecv = recv;
        api.ncclAllGather = allGather;
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
    fn run(g: *const cuda.tp_group.Group, peers: warm.Peers) !void {
        try warm.post(g, .{ .d = undefined, .handle = null }, 0x4000, peers);
    }
};

const posting = [_]warm.Peers{ .every, .neighbours };

test "GLM warm peers: ring has only neighbours, direct has every peer, all postings match and buffers stay distinct" {
    for (2..9) |world| for (posting) |peers| {
        const ring = peers == .neighbours;
        var sends: [8][8]u8 = @splat(@splat(0));
        var recvs: [8][8]u8 = @splat(@splat(0));
        for (0..world) |rank| {
            const g = Fake.group(rank, world);
            Fake.reset();
            try Fake.run(&g, peers);
            const n = if (ring) @min(world - 1, 2) else world - 1;
            try t.expectEqual(2 + 2 * n, Fake.count);
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

test "GLM warm peers: gather posts no send or receive, one all-gather of every rank's 4 MiB into the buffer's head" {
    for (2..9) |world| for (0..world) |rank| {
        const g = Fake.group(rank, world);
        Fake.reset();
        try Fake.run(&g, .gather);
        try t.expectEqual(@as(usize, 1), Fake.count);
        const c = Fake.calls[0];
        try t.expectEqual(Fake.Kind.gather, c.kind);
        try t.expectEqual(@as(usize, warm.bytes), c.bytes);
        try t.expectEqual(@as(u64, 0x4000 + rank * warm.bytes), c.pointer);
        try t.expectEqual(@as(u64, 0x4000), c.to);
        Fake.reset();
        Fake.fail = 0;
        try t.expectError(error.NcclFailed, Fake.run(&g, .gather));
        try t.expectEqual(@as(usize, 1), Fake.count);
    };
}

test "GLM warm peers: failure at every posting closes only an opened group and sends nothing after the failure" {
    for (2..9) |world| for (0..world) |rank| for (posting) |peers| {
        const g = Fake.group(rank, world);
        Fake.reset();
        try Fake.run(&g, peers);
        const total = Fake.count;
        const clean = Fake.calls;
        for (0..total) |fail| {
            Fake.reset();
            Fake.fail = fail;
            try t.expectError(error.NcclFailed, Fake.run(&g, peers));
            const posted = clean[fail].kind == .send or clean[fail].kind == .recv;
            try t.expectEqual(fail + 1 + @intFromBool(posted), Fake.count);
            for (Fake.calls[0 .. fail + 1], clean[0 .. fail + 1]) |got, wanted| try t.expectEqualDeep(wanted, got);
            if (posted) try t.expectEqual(Fake.Kind.end, Fake.calls[Fake.count - 1].kind);
        }
        Fake.reset();
        Fake.fail = 1;
        Fake.end_fails = true;
        try t.expectError(error.NcclFailed, Fake.run(&g, peers));
        try t.expectEqual(@as(usize, 3), Fake.count);
    };
}

test "GLM warm peers: a split warm-up of four ranks or more that NCCL fails at its group's end refuses split by name first; fewer ranks, ring and gather do not" {
    for (2..9) |world| for (0..world) |rank| for ([_]warm.Peers{ .every, .neighbours, .gather }) |peers| {
        const g = Fake.group(rank, world);
        Fake.reset();
        warm.refused_rank = null;
        try Fake.run(&g, peers);
        try t.expectEqual(@as(?usize, null), warm.refused_rank);
        Fake.reset();
        if (peers == .gather) Fake.fail = 0 else Fake.end_fails = true;
        try t.expectError(error.NcclFailed, Fake.run(&g, peers));
        try t.expectEqual(if (peers == .every and world >= 4) @as(?usize, rank) else null, warm.refused_rank);
    };
    warm.refused_rank = null;
}

test "GLM warm peers: one rank does nothing and malformed placement refuses before NCCL" {
    Fake.reset();
    var g = Fake.group(0, 1);
    for ([_]warm.Peers{ .every, .neighbours, .gather }) |peers| try Fake.run(&g, peers);
    try t.expectEqual(@as(usize, 0), Fake.count);
    g.world = 0;
    try t.expectError(error.BadWorld, Fake.run(&g, .neighbours));
    try t.expectError(error.BadWorld, Fake.run(&g, .gather));
    g.world = 2;
    g.rank = 2;
    try t.expectError(error.BadWorld, Fake.run(&g, .every));
    try t.expectError(error.BadWorld, Fake.run(&g, .gather));
    try t.expectEqual(@as(usize, 0), Fake.count);
}

test "GLM warm peers: the prefill selector picks the warm-up, ring its neighbours, gather its all-gather, split every peer" {
    for ([_]?[]const u8{ null, "split", "gather", "ring" }) |mode| {
        Fake.mode = mode;
        var why: []const u8 = "";
        const settings = try ranks.Settings.read(Fake.get, &why);
        const want: warm.Peers = if (mode == null or std.mem.eql(u8, mode.?, "split")) .every else if (std.mem.eql(u8, mode.?, "ring")) .neighbours else .gather;
        const peers = warm.Peers.of(settings.split, settings.ring);
        try t.expectEqual(want, peers);
        const g = Fake.group(0, 4);
        Fake.reset();
        try Fake.run(&g, peers);
        try t.expectEqual(@as(usize, switch (want) {
            .every => 8,
            .neighbours => 6,
            .gather => 1,
        }), Fake.count);
    }
    Fake.mode = null;
}

test {
    _ = @import("cuda_warm_timeout_test.zig");
}
