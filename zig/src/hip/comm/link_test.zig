//! The link's join and startup steps over loopback TCP, every rank a thread: what each rank hears.

const std = @import("std");
const link = @import("link.zig");
const rccl = @import("rccl.zig");
const agree = @import("../agree.zig");
const Policy = @import("../policy.zig").Policy;
const collective = @import("../collective.zig");
const Startup = @import("../startup.zig").Startup;
const loopback = @import("loopback.zig");
const hello = loopback.hello;
const posix = @import("sys").posix;
const auth = @import("sys").rank_auth;

/// A rank of a test group: what it says at join, and what it got.
const Rank = struct {
    hello: link.Hello,
    port: u16,
    got: ?link.Link = null,
    err: ?link.Error = null,
    fault: agree.Fault = .{},
    words: [Policy.word_count]u32 = @splat(0),
    secret: auth.Secret = .none,
    /// After joining: the startup steps it takes, and their verdicts; `leave` closes its link instead.
    steps: []const agree.Step = &.{},
    verdicts: [4]agree.Verdict = undefined,
    taken: usize = 0,
    leave: bool = false,

    fn run(r: *Rank) void {
        const id: rccl.UniqueId = .{ .internal = @splat(7) };
        const policy: [Policy.word_count]u32 = @splat(@intCast(100 + r.hello.rank));
        const out = link.Link.open(std.testing.io, "127.0.0.1", r.port, id, r.hello, policy, &r.fault, &r.secret) catch |err| {
            r.err = err;
            return;
        };
        r.got = out[0];
        r.words = out[2];
        if (r.leave) {
            r.got.?.close();
            r.got = null;
            return;
        }
        for (r.steps) |s| {
            r.verdicts[r.taken] = r.got.?.agree(s, .least) catch |err| {
                r.err = err;
                return;
            };
            r.taken += 1;
        }
    }
};

/// Runs every rank of `ranks` on its own thread, rank 0 first, and waits for all.
fn group(ranks: []Rank) !void {
    loopback.ignorePipe();
    const port = try loopback.freePort();
    var threads: [4]std.Thread = undefined;
    for (ranks, 0..) |*r, i| {
        r.port = port;
        threads[i] = try std.Thread.spawn(.{}, Rank.run, .{r});
    }
    for (threads[0..ranks.len]) |t| t.join();
    for (ranks) |*r| if (r.got) |*l| l.close();
}

test "ranks that match join, and every follower runs rank 0's policy words" {
    var ranks = [_]Rank{
        .{ .hello = hello(0, 3), .port = 0 },
        .{ .hello = hello(1, 3), .port = 0 },
        .{ .hello = hello(2, 3), .port = 0 },
    };
    try group(&ranks);
    for (ranks) |r| try std.testing.expect(r.err == null);
    for (ranks) |r| try std.testing.expectEqual(@as(u32, 100), r.words[0]);
}

test "a rank started with another world size is refused on every rank before RCCL starts" {
    var ranks = [_]Rank{ .{ .hello = hello(0, 2), .port = 0 }, .{ .hello = hello(1, 4), .port = 0 } };
    try group(&ranks);
    for (ranks) |r| {
        try std.testing.expectEqual(@as(?link.Error, error.RanksDiffer), r.err);
        try std.testing.expectEqual(agree.Fault{ .field = .world, .rank = 1 }, r.fault);
    }
}

test "a rank with other weights is refused, and so is the rank that matched before it" {
    var other = hello(2, 3);
    other.ident.weights[31] = 9;
    var ranks = [_]Rank{
        .{ .hello = hello(0, 3), .port = 0 },
        .{ .hello = hello(1, 3), .port = 0 },
        .{ .hello = other, .port = 0 },
    };
    try group(&ranks);
    for (ranks) |r| try std.testing.expectEqual(@as(?link.Error, error.RanksDiffer), r.err);
    try std.testing.expectEqual(agree.Fault{ .field = .weights, .rank = 2 }, ranks[0].fault);
    try std.testing.expectEqual(ranks[0].fault, ranks[2].fault);
}

test "hellos differ field by field in a fixed order" {
    const lead = hello(0, 2);
    var taken = [_]bool{ true, false };
    try std.testing.expect(link.differs(lead, hello(1, 2), &taken) == null);
    var h = hello(1, 2);
    h.gpu = 12;
    try std.testing.expectEqual(agree.Field.gpu, link.differs(lead, h, &taken).?);
    h.world = 4;
    try std.testing.expectEqual(agree.Field.world, link.differs(lead, h, &taken).?);
    h = hello(1, 2);
    h.ident.config[0] = 0;
    try std.testing.expectEqual(agree.Field.config, link.differs(lead, h, &taken).?);
    h = hello(1, 2);
    h.rccl = 22100;
    try std.testing.expectEqual(agree.Field.rccl, link.differs(lead, h, &taken).?);
    h = hello(1, 2);
    h.ident.program[5] = 0;
    try std.testing.expectEqual(agree.Field.program, link.differs(lead, h, &taken).?);
    h = hello(1, 2);
    h.layout = 1;
    try std.testing.expectEqual(agree.Field.layout, link.differs(lead, h, &taken).?);
    taken[1] = true;
    try std.testing.expectEqual(agree.Field.rank, link.differs(lead, hello(1, 2), &taken).?);
}

test "startup steps: the least of each value on every rank, then one rank's refusal on every rank" {
    const ok0 = [_]agree.Step{ .ok(.{ 8, 30 }), .ok(.{ 1, 1 }) };
    const ok1 = [_]agree.Step{ .ok(.{ 6, 40 }), .ok(.{ 1, 1 }) };
    const no2 = [_]agree.Step{ .ok(.{ 7, 20 }), .refuse("--parallel 4 does not fit") };
    var ranks = [_]Rank{
        .{ .hello = hello(0, 3), .port = 0, .steps = &ok0 },
        .{ .hello = hello(1, 3), .port = 0, .steps = &ok1 },
        .{ .hello = hello(2, 3), .port = 0, .steps = &no2 },
    };
    try group(&ranks);
    for (ranks) |r| {
        try std.testing.expect(r.err == null);
        try std.testing.expectEqual([2]u64{ 6, 20 }, r.verdicts[0].values);
        try std.testing.expectEqual(@as(u32, 1), r.verdicts[1].refused);
        try std.testing.expectEqual(@as(u32, 2), r.verdicts[1].rank);
        try std.testing.expectEqualStrings("--parallel 4 does not fit", r.verdicts[1].reason());
    }
}

test "a rank that leaves before a step is that step's refusal on every rank that stays" {
    const one = [_]agree.Step{.ok(.{ 1, 1 })};
    var ranks = [_]Rank{
        .{ .hello = hello(0, 3), .port = 0, .steps = &one },
        .{ .hello = hello(1, 3), .port = 0, .steps = &one },
        .{ .hello = hello(2, 3), .port = 0, .leave = true },
    };
    try group(&ranks);
    for (ranks[0..2]) |r| {
        try std.testing.expect(r.err == null);
        try std.testing.expectEqual(@as(u32, 1), r.verdicts[0].refused);
        try std.testing.expectEqual(@as(u32, 2), r.verdicts[0].rank);
        try std.testing.expectEqualStrings("its link to rank 0 closed", r.verdicts[0].reason());
    }
}

/// A family's startup on each rank: rank 0's window, admission (`refuse`: the rank that cannot), the streams that fit.
const Families = struct {
    refuse: ?usize = null,
    window: [3]u64 = @splat(0),
    fit: [3]u64 = @splat(0),
    problem: [3][]const u8 = @splat(""),
    text: [3][256]u8 = undefined,

    fn start(f: *Families, rank: usize, l: collective.Link) void {
        var fba: std.heap.FixedBufferAllocator = .init(&f.text[rank]);
        var problem: []const u8 = "";
        f.steps(fba.allocator(), rank, l, &problem) catch {};
        f.problem[rank] = problem;
    }

    fn steps(f: *Families, a: std.mem.Allocator, rank: usize, l: collective.Link, problem: *[]const u8) !void {
        var s: Startup = .{ .link = l };
        f.window[rank] = (try s.together(a, .{ 4096 + rank, 0 }, .first, problem))[0];
        const why = "rank {d} cannot hold a {d}-token window";
        if (f.refuse == rank) return s.refuseAll(a, problem, why, .{ rank, f.window[rank] });
        _ = try s.together(a, .{ 0, 0 }, .first, problem);
        f.fit[rank] = (try s.together(a, .{ 8 - rank, 0 }, .least, problem))[0];
    }
};

test "a family's startup: rank 0's window and the least fit on every rank, or one rank's refusal named on all" {
    var ok: Families = .{};
    try loopback.run(3, &ok, Families.start);
    for (0..3) |r| {
        try std.testing.expectEqual(@as(u64, 4096), ok.window[r]);
        try std.testing.expectEqual(@as(u64, 6), ok.fit[r]);
        try std.testing.expectEqualStrings("", ok.problem[r]);
    }
    var no: Families = .{ .refuse = 2 };
    try loopback.run(3, &no, Families.start);
    const why = "rank 2 cannot hold a 4096-token window";
    try std.testing.expectEqualStrings(why, no.problem[2]);
    for (0..2) |r| {
        try std.testing.expectEqual(@as(u64, 0), no.fit[r]);
        try std.testing.expectEqualStrings("rank 2 of the group refused: " ++ why, no.problem[r]);
    }
}

/// One change every rank makes together: `fail` is the rank whose change fails, `leave` the rank that is gone.
const Changes = struct {
    fail: ?usize = null,
    leave: ?usize = null,
    all: [3]?bool = @splat(null),

    fn make(c: *Changes, rank: usize, l: collective.Link) void {
        if (c.leave == rank) return;
        const mine: anyerror!void = if (c.fail == rank) error.OutOfMemory else {};
        c.all[rank] = collective.everyRank(l, mine);
    }
};

test "a change every rank makes: kept when all went through, else undone on every rank, a rank that left too" {
    var ok: Changes = .{};
    try loopback.run(3, &ok, Changes.make);
    for (ok.all) |a| try std.testing.expectEqual(@as(?bool, true), a);
    var one: Changes = .{ .fail = 1 };
    try loopback.run(3, &one, Changes.make);
    for (one.all) |a| try std.testing.expectEqual(@as(?bool, false), a);
    var gone: Changes = .{ .leave = 2 };
    try loopback.run(3, &gone, Changes.make);
    for (gone.all[0..2]) |a| try std.testing.expectEqual(@as(?bool, false), a);
    try std.testing.expect(collective.everyRank(null, {}));
    try std.testing.expect(!collective.everyRank(null, error.OutOfMemory));
}

/// A connection that is not a rank: it reaches rank 0's port, sends `bytes`, and returns how many bytes came back.
fn stranger(port: u16, bytes: []const u8) !usize {
    var addr: posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = std.mem.nativeToBig(u32, 0x7f000001) };
    var tries: usize = 0;
    const fd: posix.socket_t = while (true) : (tries += 1) {
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
        if (posix.errno(posix.system.connect(@intCast(rc), @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) == .SUCCESS) break @intCast(rc);
        _ = posix.system.close(@intCast(rc));
        if (tries >= 2000) return error.ConnectFailed;
        std.Io.sleep(std.testing.io, .fromMilliseconds(5), .awake) catch {};
    };
    defer _ = posix.system.close(fd);
    if (posix.errno(posix.system.write(fd, bytes.ptr, bytes.len)) != .SUCCESS) return error.LinkClosed;
    var got: usize = 0;
    var buf: [256]u8 = undefined;
    while (true) {
        const rc = posix.system.read(fd, &buf, buf.len);
        if (posix.errno(rc) != .SUCCESS or rc == 0) return got; // closed, or reset with its hello unread
        got += @intCast(rc);
    }
}

test "security: a stranger that says a rank's hello first hears nothing and takes no place" {
    loopback.ignorePipe();
    const key = try auth.Secret.fromBytes("the group's rank secret for the tests");
    var ranks = [_]Rank{
        .{ .hello = hello(0, 2), .port = try loopback.freePort(), .secret = key },
        .{ .hello = hello(1, 2), .port = 0, .secret = key },
    };
    ranks[1].port = ranks[0].port;
    const lead = try std.Thread.spawn(.{}, Rank.run, .{&ranks[0]});
    // rank 1's own hello, word for word: before the proof it took rank 1's place and was sent RCCL's id
    try std.testing.expectEqual(@as(usize, 0), try stranger(ranks[0].port, std.mem.asBytes(&hello(1, 2))));
    ranks[1].run();
    lead.join();
    for (&ranks) |*r| {
        try std.testing.expectEqual(@as(?link.Error, null), r.err);
        r.got.?.close();
    }
    try std.testing.expectEqual(@as(u32, 100), ranks[1].words[0]);
}

test "security: a rank with another key or with none is refused by name, and rank 0 still takes its own" {
    loopback.ignorePipe();
    const key = try auth.Secret.fromBytes("the group's rank secret for the tests");
    var lead: Rank = .{ .hello = hello(0, 2), .port = try loopback.freePort(), .secret = key };
    const thread = try std.Thread.spawn(.{}, Rank.run, .{&lead});
    var other: Rank = .{ .hello = hello(1, 2), .port = lead.port, .secret = try auth.Secret.fromBytes("another group's secret for the tests") };
    other.run();
    try std.testing.expectEqual(@as(?link.Error, error.RankAuthWrongKey), other.err);
    var bare: Rank = .{ .hello = hello(1, 2), .port = lead.port };
    bare.run();
    try std.testing.expectEqual(@as(?link.Error, error.RankAuthPeerHasKey), bare.err);
    var mine: Rank = .{ .hello = hello(1, 2), .port = lead.port, .secret = key };
    mine.run();
    thread.join();
    try std.testing.expectEqual(@as(?link.Error, null), mine.err);
    try std.testing.expectEqual(@as(?link.Error, null), lead.err);
    mine.got.?.close();
    lead.got.?.close();
}

/// A host on the path between rank 1 and rank 0: it passes every byte, keeps what rank 0 sent, and can change a bit.
const Relay = struct {
    server: posix.socket_t,
    to: u16,
    seen: [1 << 14]u8 = undefined,
    len: usize = 0,
    tamper: std.atomic.Value(bool) = .init(false),

    fn listen(port: u16) !posix.socket_t {
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
        var addr: posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = std.mem.nativeToBig(u32, 0x7f000001) };
        if (posix.errno(posix.system.bind(@intCast(rc), @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.BindFailed;
        if (posix.errno(posix.system.listen(@intCast(rc), 1)) != .SUCCESS) return error.BindFailed;
        return @intCast(rc);
    }

    fn run(r: *Relay) void {
        const down: posix.socket_t = @intCast(posix.system.accept(r.server, null, null));
        defer _ = posix.system.close(down);
        var addr: posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, r.to), .addr = std.mem.nativeToBig(u32, 0x7f000001) };
        const up: posix.socket_t = @intCast(posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0));
        defer _ = posix.system.close(up);
        while (posix.errno(posix.system.connect(up, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) {
            std.Io.sleep(std.testing.io, .fromMilliseconds(5), .awake) catch {};
        }
        var fds = [_]posix.pollfd{ .{ .fd = down, .events = posix.POLL.IN, .revents = 0 }, .{ .fd = up, .events = posix.POLL.IN, .revents = 0 } };
        var buf: [4096]u8 = undefined;
        while (true) {
            _ = posix.poll(&fds, -1) catch return;
            for (fds, 0..) |f, i| {
                if (f.revents == 0) continue;
                const rc = posix.system.read(f.fd, &buf, buf.len);
                if (posix.errno(rc) != .SUCCESS or rc == 0) return;
                const n: usize = @intCast(rc);
                if (i == 1) {
                    // the last byte it holds is the end of a record's tag
                    if (r.tamper.swap(false, .acq_rel)) buf[n - 1] ^= 1;
                    @memcpy(r.seen[r.len..][0..n], buf[0..n]);
                    r.len += n;
                }
                var at: usize = 0;
                while (at < n) {
                    const w = posix.system.write(fds[1 - i].fd, buf[at..].ptr, n - at);
                    if (posix.errno(w) != .SUCCESS) return;
                    at += @intCast(w);
                }
            }
        }
    }
};

/// Rank 0 of a pair: it joins, waits to be told, and sends `words`.
const Sender = struct {
    port: u16,
    secret: auth.Secret,
    words: []const u32,
    go: std.atomic.Value(bool) = .init(false),
    err: ?link.Error = null,

    fn run(s: *Sender) void {
        var fault: agree.Fault = .{};
        const out = link.Link.open(std.testing.io, "127.0.0.1", s.port, .{ .internal = @splat(7) }, hello(0, 2), @splat(100), &fault, &s.secret) catch |err| {
            s.err = err;
            return;
        };
        var l = out[0];
        defer l.close();
        while (!s.go.load(.acquire)) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        l.send(s.words) catch |err| {
            s.err = err;
        };
    }
};

/// Rank 1 joins rank 0 through a relay and reads one message; `tamper` has the relay change a bit of it.
fn throughRelay(words: []const u32, tamper: bool, relay: *Relay, got: *std.ArrayList(u32)) !void {
    loopback.ignorePipe();
    const key = try auth.Secret.fromBytes("the group's rank secret");
    var sender: Sender = .{ .port = try loopback.freePort(), .secret = key, .words = words };
    const relay_port = try loopback.freePort();
    relay.* = .{ .server = try Relay.listen(relay_port), .to = sender.port };
    defer _ = posix.system.close(relay.server);
    const lead = try std.Thread.spawn(.{}, Sender.run, .{&sender});
    const path = try std.Thread.spawn(.{}, Relay.run, .{relay});
    var fault: agree.Fault = .{};
    const out = try link.Link.open(std.testing.io, "127.0.0.1", relay_port, .{ .internal = @splat(7) }, hello(1, 2), @splat(101), &fault, &key);
    var mine = out[0];
    if (tamper) relay.tamper.store(true, .release);
    sender.go.store(true, .release);
    const result = mine.recv(std.testing.allocator, got);
    mine.close();
    lead.join();
    path.join();
    try std.testing.expectEqual(@as(?link.Error, null), sender.err);
    return result;
}

test "security: rank 0's words cross the link sealed, and a bit changed on the way closes it" {
    const words = [_]u32{ 0x51525354, 0x61626364, 0x51525354, 0x71727374, 17, 4021, 9, 88213 };
    var got: std.ArrayList(u32) = .empty;
    defer got.deinit(std.testing.allocator);
    var relay: Relay = undefined;
    try throughRelay(&words, false, &relay, &got);
    try std.testing.expectEqualSlices(u32, &words, got.items);
    // a host on the path saw the key proof and then records: none of the words, and not the policy words either
    const seen = relay.seen[0..relay.len];
    try std.testing.expect(seen.len > words.len * 4);
    for ([_]u32{ 0x51525354, 0x61626364, 0x71727374, 88213 }) |w| try std.testing.expect(std.mem.indexOf(u8, seen, std.mem.asBytes(&w)) == null);
    try std.testing.expectError(error.LinkClosed, throughRelay(&words, true, &relay, &got));
}
