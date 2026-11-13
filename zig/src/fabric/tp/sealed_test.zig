//! Two ranks with a relay on the path between them: what it reads of a keyed group, and what a changed byte does.
const std = @import("std");
const sock = @import("sock.zig");
const fate = @import("fate.zig");
const bootstrap = @import("bootstrap.zig");
const planlink = @import("planlink.zig");
const auth = @import("sys").rank_auth;
const Fate = fate.Fate;

const settings = "tp settings: graph cache on, world 2";
const plan_text = "plan: prompt token ids 17 4021 9";
const plan_bytes: [32]u8 = plan_text.*;
const plan_words: [4]i64 = @bitCast(plan_bytes);
const reason = "rank 1 failed in prefill";
const seconds = std.time.ns_per_s;

var exits: std.atomic.Value(u32) = .init(0);

fn recordExit(_: u8) void {
    _ = exits.fetchAdd(1, .acq_rel);
}

/// One bit to flip on its way: connection 0 is the control link and 1 the plan link, `down` is rank 0 to rank 1.
const Change = struct { conn: usize, down: bool, at: usize };

/// A host on the path: it carries both of rank 1's connections to rank 0 and keeps every byte of each direction.
const Relay = struct {
    listener: sock.Fd,
    upstream: u16,
    change: ?Change = null,
    seen: [2][2][]u8 = undefined,
    len: [2][2]usize = @splat(@splat(0)),
    err: ?anyerror = null,

    fn run(r: *Relay) void {
        r.carry() catch |e| {
            r.err = e;
        };
    }

    fn carry(r: *Relay) !void {
        var ends: [2][2]sock.Fd = undefined;
        var pumps: [2][2]std.Thread = undefined;
        for (0..2) |conn| {
            const until = sock.nowNs() + 5 * seconds;
            ends[conn][0] = try sock.accept(r.listener, until);
            ends[conn][1] = try sock.connect("localhost", r.upstream, until);
            for (0..2) |down| pumps[conn][down] = try std.Thread.spawn(.{}, pump, .{ r, conn, down, ends[conn][down], ends[conn][1 - down] });
        }
        for (pumps) |pair| for (pair) |t| t.join();
        for (ends) |pair| for (pair) |fd| sock.close(fd);
    }

    fn pump(r: *Relay, conn: usize, down: usize, from: sock.Fd, to: sock.Fd) void {
        var buf: [4096]u8 = undefined;
        var total: usize = 0;
        while (true) {
            const n = std.c.recv(from, &buf, buf.len, 0);
            if (n <= 0) break;
            const got = buf[0..@intCast(n)];
            if (r.change) |c| if (c.conn == conn and @intFromBool(c.down) == down and c.at >= total and c.at < total + got.len) {
                got[c.at - total] ^= 1;
            };
            total += got.len;
            const kept = @min(got.len, r.seen[conn][down].len - r.len[conn][down]);
            @memcpy(r.seen[conn][down][r.len[conn][down]..][0..kept], got[0..kept]);
            r.len[conn][down] += kept;
            sock.sendAll(to, got) catch break;
        }
        _ = std.c.shutdown(to, std.posix.SHUT.WR);
    }

    fn read(r: *const Relay, words: []const u8) bool {
        for (0..2) |conn| for (0..2) |down| {
            if (std.mem.indexOf(u8, r.seen[conn][down][0..r.len[conn][down]], words) != null) return true;
        };
        return false;
    }
};

const Follower = struct {
    plan: bool = false,
    err: ?anyerror = null,

    fn run(port: u16, secret: auth.Secret, f: *Fate, out: *Follower) void {
        out.join(port, secret, f) catch |e| {
            out.err = e;
        };
    }

    fn join(out: *Follower, port: u16, secret: auth.Secret, f: *Fate) !void {
        var b = try bootstrap.Bootstrap.open(.{ .world = 2, .rank = 1, .link_port = port, .connect_timeout_ns = 5 * seconds, .secret = secret });
        defer b.close();
        try b.agree(settings, "settings");
        var plan = planlink.PlanLink.init(1, 2, b.releasePlan());
        defer plan.close();
        var words: [16]i64 = undefined;
        out.plan = std.mem.eql(i64, try plan.recv(&words), &plan_words);
        f.* = Fate.init(.{ .world = 2, .rank = 1, .grace_ns = 20 * std.time.ns_per_ms, .report_ns = 20 * std.time.ns_per_ms }, b.release());
        f.exit_fn = recordExit;
        try f.start();
    }
};

/// Rank 1's side of a meeting, and whether the relay read any of the settings, the plan or the failure's reason.
const Met = struct { follower: Follower, read: bool };

/// The two ranks through `relay`: settings agreed, a plan sent, then rank 1 fails.
fn meet(secret: auth.Secret, relay: *Relay) !Met {
    const gpa = std.testing.allocator;
    for (&relay.seen) |*conn| for (conn) |*s| {
        s.* = try gpa.alloc(u8, 64 * 1024);
    };
    defer for (relay.seen) |conn| for (conn) |s| gpa.free(s);
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    const front = try bootstrap.testListener();
    defer sock.close(front.fd);
    relay.listener = front.fd;
    relay.upstream = l.port;
    const carrier = try std.Thread.spawn(.{}, Relay.run, .{relay});
    var fates: [2]Fate = undefined;
    var follower: Follower = .{};
    const rank1 = try std.Thread.spawn(.{}, Follower.run, .{ front.port, secret, &fates[1], &follower });
    exits.store(0, .release);
    var b = try bootstrap.Bootstrap.acceptWith(l.fd, 2, 5 * seconds, secret);
    try b.agree(settings, "settings");
    var plan = planlink.PlanLink.init(0, 2, b.releasePlan());
    try plan.send(&plan_words);
    rank1.join();
    if (follower.err == null) {
        fates[0] = Fate.init(.{ .world = 2, .rank = 0, .grace_ns = 20 * std.time.ns_per_ms, .report_ns = 20 * std.time.ns_per_ms }, b.release());
        fates[0].exit_fn = recordExit;
        try fates[0].start();
        fates[1].fatal("prefill", error.OutOfMemory);
        const t0 = sock.nowNs();
        while (exits.load(.acquire) < 2 and sock.nowNs() - t0 < 5 * seconds) sock.sleepNs(std.time.ns_per_ms);
        for (&fates) |*f| f.close();
        if (exits.load(.acquire) != 2) follower.err = error.RanksStayedUp;
        if (std.mem.indexOf(u8, fates[0].reason(), reason) == null) follower.err = error.NoReason;
    }
    plan.close();
    b.close();
    carrier.join();
    if (relay.err) |e| return e;
    return .{ .follower = follower, .read = relay.read(settings) or relay.read(plan_text) or relay.read(reason) };
}

test "security: a host on the path reads none of a keyed group's settings, plans or failure reasons" {
    var relay: Relay = .{ .listener = -1, .upstream = 0 };
    const keyed = try meet(try auth.Secret.fromBytes("the group's rank secret for the tests"), &relay);
    try std.testing.expectEqual(@as(?anyerror, null), keyed.follower.err);
    try std.testing.expect(keyed.follower.plan);
    try std.testing.expect(!keyed.read);
    // the same ranks with no key, as one host runs them: the relay reads them, so the check above sees what crosses
    var open: Relay = .{ .listener = -1, .upstream = 0 };
    const plain = try meet(.none, &open);
    try std.testing.expectEqual(@as(?anyerror, null), plain.follower.err);
    try std.testing.expect(plain.follower.plan);
    try std.testing.expect(plain.read);
}

test "security: a byte changed on a keyed group's plan link ends the link, and no plan is taken from it" {
    // rank 0's first record after its 76-byte side of the proof: its length, its sealed bytes, then its tag
    for ([_]usize{ 0, 5, 12 }) |offset| {
        var relay: Relay = .{ .listener = -1, .upstream = 0, .change = .{ .conn = 1, .down = true, .at = auth.challenge_len + offset } };
        const met = try meet(try auth.Secret.fromBytes("the group's rank secret for the tests"), &relay);
        try std.testing.expectEqual(@as(?anyerror, error.PeerClosed), met.follower.err);
        try std.testing.expect(!met.follower.plan);
    }
}

test "security: a new socket whose number a sealed one had carries its bytes as they are" {
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    const until = sock.nowNs() + 5 * seconds;
    const a = try sock.connect("localhost", l.port, until);
    const b = try sock.accept(l.fd, until);
    defer sock.close(b);
    const session: auth.Key = @splat(7);
    try sock.seal(a, &session, true);
    try std.testing.expect(sock.isSealed(a));
    _ = std.c.close(a); // closed behind the helpers' back: its seal is still listed under its number
    const again = try sock.connect("localhost", l.port, until);
    defer sock.close(again);
    const peer = try sock.accept(l.fd, until);
    defer sock.close(peer);
    try std.testing.expect(!sock.isSealed(again));
    try sock.sendAll(again, "hello");
    var got: [5]u8 = undefined;
    try sock.recvExact(peer, &got, until);
    try std.testing.expectEqualStrings("hello", &got);
}
