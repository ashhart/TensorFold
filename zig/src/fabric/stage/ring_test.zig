//! Rings of 2 and 3 ranks on loopback threads: rows go round exact and in order, and every refusal fires as planted.
const std = @import("std");
const ring = @import("ring.zig");
const net = @import("net.zig");
const t = std.testing;

const most = 3;
const fp_a: [32]u8 = @splat(0xA5);
const key_a: [32]u8 = @splat(0x3C);

/// Every rank's listener first (kernel-picked ports), then their options.
const Setup = struct {
    peers: [most]ring.Peer,
    listeners: [most]net.Fd,
    opts: [most]ring.Options,

    fn init(s: *Setup, world: u32) !void {
        for (0..world) |i| {
            s.listeners[i] = try net.listen("127.0.0.1", 0);
            s.peers[i] = .{ .host = "127.0.0.1", .port = try net.port(s.listeners[i]) };
        }
        for (0..world) |i| s.opts[i] = .{ .rank = @intCast(i), .world = world, .peers = s.peers[0..world], .fingerprint = fp_a, .key = key_a, .timeout_ns = 5 * std.time.ns_per_s };
    }
};

const Joined = struct { r: ?ring.Ring = null, err: ?anyerror = null };

fn isErr(got: ?anyerror, want: anyerror) bool {
    return if (got) |e| e == want else false;
}

fn joinOne(o: ring.Options, l: net.Fd, out: *Joined) void {
    out.r = ring.Ring.join(o, l) catch |e| {
        out.err = e;
        return;
    };
}

/// Joins `world` ranks at once; each slot holds its ring or its error.
fn joinAll(s: *Setup, world: u32, out: []Joined) !void {
    var th: [most]std.Thread = undefined;
    for (0..world) |i| th[i] = try std.Thread.spawn(.{}, joinOne, .{ s.opts[i], s.listeners[i], &out[i] });
    for (th[0..world]) |x| x.join();
}

/// One round on rank `rank` of the ring: rank 0 sends `rows` and receives what the last rank sends back.
fn stage(r: *ring.Ring, round: u64, rows: []const u8, out: []u8, delay_ms: u64) !void {
    if (r.rank == 0) {
        try r.send(.plan, round, 0, &.{std.mem.asBytes(&round)});
        try r.send(.rows, round, @intCast(rows.len / 8), &.{ rows[0 .. rows.len / 2], rows[rows.len / 2 ..] });
        _ = try r.recv(.rows, round, out);
        return;
    }
    var plan: [8]u8 = undefined;
    _ = try r.recv(.plan, round, &plan);
    const h = try r.recv(.rows, round, out);
    if (delay_ms > 0) net.sleepNs(delay_ms * std.time.ns_per_ms);
    for (out[0..@intCast(h.bytes)]) |*b| b.* +%= 1; // a stage's work: every byte plus one
    if (r.rank + 1 < r.world) try r.send(.plan, round, 0, &.{&plan});
    try r.send(.rows, round, h.rows, &.{out[0..@intCast(h.bytes)]});
}

fn follower(r: *ring.Ring, rounds: u64, bytes: usize, delay_ms: u64, err: *?anyerror) void {
    var buf: [4096]u8 = undefined;
    var round: u64 = 1;
    while (round <= rounds) : (round += 1) stage(r, round, buf[0..bytes], &buf, delay_ms) catch |e| {
        err.* = e;
        r.close();
        return;
    };
}

fn roundTrip(world: u32, delay_ms: u64) !void {
    var s: Setup = undefined;
    try s.init(world);
    var j: [most]Joined = @splat(.{});
    try joinAll(&s, world, &j);
    for (j[0..world]) |x| try t.expect(x.err == null);
    defer for (j[0..world]) |*x| x.r.?.close();
    var errs: [most]?anyerror = @splat(null);
    var th: [most]std.Thread = undefined;
    for (1..world) |i| th[i] = try std.Thread.spawn(.{}, follower, .{ &j[i].r.?, 3, 0, delay_ms, &errs[i] });
    var rows: [4096]u8 = undefined;
    var back: [4096]u8 = undefined;
    for (1..4) |round| {
        for (&rows, 0..) |*b, i| b.* = @truncate(i * 7 + round);
        try stage(&j[0].r.?, round, &rows, &back, 0);
        for (rows, back) |a, b| try t.expectEqual(a +% @as(u8, @intCast(world - 1)), b);
    }
    for (th[1..world]) |x| x.join();
    for (errs[1..world]) |e| try t.expect(e == null);
}

test "stage ring: two and three ranks carry a plan and split rows round in order, every byte exact" {
    try roundTrip(2, 0);
    try roundTrip(3, 0);
}

test "stage ring: a slow middle stage only delays the ring, the bytes are the same" {
    try roundTrip(3, 30);
}

test "stage ring: another fingerprint or another world is refused at the hello" {
    var s: Setup = undefined;
    try s.init(2);
    s.opts[1].fingerprint[0] ^= 1;
    var j: [most]Joined = @splat(.{});
    try joinAll(&s, 2, &j);
    for (j[0..2]) |*x| if (x.r) |*r| r.close();
    try t.expect(isErr(j[0].err, error.OtherFingerprint) or isErr(j[1].err, error.OtherFingerprint));
    try s.init(2);
    s.opts[0].world = 3;
    s.opts[0].peers = &.{ s.peers[0], s.peers[1], s.peers[1] };
    j = @splat(.{});
    try joinAll(&s, 2, &j);
    for (j[0..2]) |*x| if (x.r) |*r| r.close();
    try t.expect(j[0].err != null or j[1].err != null);
}

/// Two joined ranks on loopback.
fn pair(j: *[most]Joined) !void {
    var s: Setup = undefined;
    try s.init(2);
    j.* = @splat(.{});
    try joinAll(&s, 2, j);
    for (j[0..2]) |x| try t.expect(x.err == null);
}

fn closePair(j: *[most]Joined) void {
    j[0].r.?.close();
    j[1].r.?.close();
}

test "stage ring: a rank without the key is refused by both its neighbours, in a ring of two and of three" {
    for ([_]u32{ 2, 3 }) |world| {
        var s: Setup = undefined;
        try s.init(world);
        s.opts[1].key[0] ^= 1;
        var j: [most]Joined = @splat(.{});
        try joinAll(&s, world, &j);
        for (j[0..world]) |*x| if (x.r) |*r| r.close();
        try t.expect(isErr(j[0].err, error.BadSecret) or isErr(j[0].err, error.Closed));
        try t.expect(isErr(j[1].err, error.BadSecret) or isErr(j[1].err, error.Closed));
        try t.expect(isErr(j[0].err, error.BadSecret) or isErr(j[1].err, error.BadSecret));
        if (world == 3) try t.expect(isErr(j[2].err, error.BadSecret) or isErr(j[2].err, error.Closed));
    }
}

test "stage ring: a stranger that connects and talks first is refused before any secret byte is sent" {
    var s: Setup = undefined;
    try s.init(2);
    const stranger = try net.connect("127.0.0.1", s.peers[1].port, net.nowNs() + std.time.ns_per_s);
    defer net.close(stranger);
    try net.sendAll(stranger, "GET / HTTP/1.1\r\nHost: stage-ring\r\n\r\nxxxxxxxx", net.nowNs() + std.time.ns_per_s);
    defer net.close(s.listeners[0]); // rank 0's backlog takes rank 1's connect, so rank 1 reaches its accept
    var o = s.opts[1];
    o.timeout_ns = 500 * std.time.ns_per_ms;
    try t.expectError(error.BadMagic, ring.Ring.join(o, s.listeners[1]));
}

test "stage ring: a frame out of round or of another kind is refused, and the link is broken after it" {
    var buf: [16]u8 = undefined;
    var j: [most]Joined = undefined;
    try pair(&j);
    try t.expectError(error.OutOfOrder, j[0].r.?.send(.rows, 5, 0, &.{}));
    try t.expectError(error.Broken, j[0].r.?.send(.rows, 1, 0, &.{}));
    closePair(&j);
    try pair(&j);
    try j[0].r.?.send(.rows, 1, 0, &.{"abc"});
    try t.expectError(error.OutOfOrder, j[1].r.?.recv(.rows, 2, &buf));
    try t.expectError(error.Broken, j[1].r.?.recv(.rows, 1, &buf));
    closePair(&j);
    try pair(&j);
    try j[0].r.?.send(.plan, 1, 0, &.{"x"});
    try t.expectError(error.WrongKind, j[1].r.?.recv(.rows, 1, &buf));
    closePair(&j);
}

test "stage ring: a stop ends the next rank's wait" {
    var buf: [16]u8 = undefined;
    var j: [most]Joined = undefined;
    try pair(&j);
    defer closePair(&j);
    try j[0].r.?.send(.rows, 1, 2, &.{"ab"});
    _ = try j[1].r.?.recv(.rows, 1, &buf);
    j[0].r.?.stop();
    try t.expectError(error.Stopped, j[1].r.?.recv(.rows, 2, &buf));
}

test "stage ring: a frame past the receiver's buffer is refused before its payload is read" {
    var s: Setup = undefined;
    try s.init(2);
    var j: [most]Joined = @splat(.{});
    try joinAll(&s, 2, &j);
    defer j[0].r.?.close();
    defer j[1].r.?.close();
    try j[0].r.?.send(.rows, 1, 1, &.{"0123456789"});
    var small: [4]u8 = undefined;
    try t.expectError(error.TooLarge, j[1].r.?.recv(.rows, 1, &small));
}

test "stage ring: a rank that fails closes its links, and both neighbours of a three-ring see it" {
    var s: Setup = undefined;
    try s.init(3);
    var j: [most]Joined = @splat(.{});
    try joinAll(&s, 3, &j);
    defer j[0].r.?.close();
    defer j[2].r.?.close();
    j[1].r.?.close(); // rank 1 dies
    var buf: [16]u8 = undefined;
    try t.expectError(error.Closed, j[2].r.?.recv(.rows, 1, &buf));
    var closed = false;
    for (0..50) |_| {
        j[0].r.?.send(.rows, 1, 0, &.{"xxxxxxxxxxxxxxxx"}) catch |e| {
            closed = isErr(e, error.Closed);
            break;
        };
    }
    try t.expect(closed);
}

test "stage ring: a ring needs two ranks, a rank inside it and one address a rank" {
    const p = [_]ring.Peer{.{ .host = "127.0.0.1", .port = 1 }};
    try t.expectError(error.BadRing, ring.Ring.bind(.{ .rank = 0, .world = 1, .peers = &p, .fingerprint = fp_a, .key = key_a }));
    try t.expectError(error.BadRing, ring.Ring.bind(.{ .rank = 2, .world = 2, .peers = &.{ p[0], p[0] }, .fingerprint = fp_a, .key = key_a }));
    try t.expectError(error.BadRing, ring.Ring.bind(.{ .rank = 0, .world = 2, .peers = &p, .fingerprint = fp_a, .key = key_a }));
}
