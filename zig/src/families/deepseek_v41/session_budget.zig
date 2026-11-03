//! `--prompt-cache-gib` as the kept-state store's resident byte limit, rank 0's sent to every rank at boot.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const tp = @import("tp");

const timeout_ns = 5 * std.time.ns_per_s;

/// Rank 0's resident limit (null: unlimited; 0: no idle state resident) and whether the variable beat the flag.
pub const Limit = struct { bytes: ?u64, env_wins: bool = false };

/// `env`: TF_DSV41_SESSION_RESIDENT_GIB's bytes, which win; `gib`: the flag (the server's checked 0 to 2^34).
pub fn resolve(env: ?u64, gib: ?f64) Limit {
    const g = gib orelse return .{ .bytes = env };
    if (env != null) return .{ .bytes = env, .env_wins = true };
    return .{ .bytes = @intFromFloat(g * (1 << 30)) };
}

/// A rank's own variable is set and is not rank 0's limit: its store would evict where rank 0's does not.
pub fn differs(own: ?u64, leader: ?u64) bool {
    const o = own orelse return false;
    return leader == null or leader.? != o;
}

/// The refusal: the variable by name, the rank and both limits.
pub fn refusal(buf: []u8, rank: usize, own: u64, leader: ?u64) []const u8 {
    var lb: [24]u8 = undefined;
    const l = if (leader) |b| std.fmt.bufPrint(&lb, "{d} bytes", .{b}) catch unreachable else "unlimited";
    return std.fmt.bufPrint(buf, "sessions: TF_DSV41_SESSION_RESIDENT_GIB on rank {d} is {d} bytes, but rank 0's resident limit is {s}; unset it on that rank or match rank 0", .{ rank, own, l }) catch buf;
}

/// Every rank's resident limit: rank 0 resolves its own and sends it; a differing follower variable refuses everywhere.
pub fn agree(plan: *tp.planlink.PlanLink, env: ?u64, gib: ?f64) !?u64 {
    if (plan.rank != 0) return follow(plan, env);
    const mine = resolve(env, gib);
    if (mine.env_wins and !quiet) std.log.scoped(.dsv41).info("sessions: --prompt-cache-gib ignored: TF_DSV41_SESSION_RESIDENT_GIB is set and wins", .{});
    if (plan.world > 1) try lead(plan, mine.bytes);
    return mine.bytes;
}

fn encode(bytes: ?u64) [2]i64 {
    return .{ @intFromBool(bytes != null), @bitCast(bytes orelse 0) };
}

fn decode(w: [2]i64) ?u64 {
    return if (w[0] == 1) @as(u64, @bitCast(w[1])) else null;
}

fn send(plan: *tp.planlink.PlanLink, peer: usize, words: []const i64, deadline: u64) !void {
    try tp.sock.sendBefore(plan.fds[peer], std.mem.sliceAsBytes(words), deadline);
}

fn recv(plan: *tp.planlink.PlanLink, peer: usize, words: []i64, deadline: u64) !void {
    try tp.sock.recvExact(plan.fds[peer], std.mem.sliceAsBytes(words), deadline);
}

fn refuse(rank: usize, own: u64, leader: ?u64) void {
    var buf: [256]u8 = undefined;
    if (!quiet) std.log.scoped(.dsv41).err("{s}", .{refusal(&buf, rank, own, leader)});
}

fn lead(plan: *tp.planlink.PlanLink, bytes: ?u64) !void {
    const deadline = tp.sock.nowNs() + timeout_ns;
    var ok = true;
    for (1..plan.world) |rank| {
        try send(plan, rank, &encode(bytes), deadline);
        var peer: [2]i64 = undefined;
        try recv(plan, rank, &peer, deadline);
        if (differs(decode(peer), bytes)) {
            refuse(rank, decode(peer).?, bytes);
            ok = false;
        }
    }
    const verdict = [_]i64{@intFromBool(ok)};
    for (1..plan.world) |rank| try send(plan, rank, &verdict, deadline);
    if (!ok) return error.SessionResidentMismatch;
}

fn follow(plan: *tp.planlink.PlanLink, env: ?u64) !?u64 {
    const deadline = tp.sock.nowNs() + timeout_ns;
    var word: [2]i64 = undefined;
    try recv(plan, 0, &word, deadline);
    const leader = decode(word);
    try send(plan, 0, &encode(env), deadline);
    var verdict: [1]i64 = undefined;
    try recv(plan, 0, &verdict, deadline);
    if (differs(env, leader)) refuse(plan.rank, env.?, leader);
    if (verdict[0] != 1 or differs(env, leader)) return error.SessionResidentMismatch;
    return leader;
}

test "prompt cache: --prompt-cache-gib 4 sets a resident limit of 4 << 30 bytes" {
    try std.testing.expectEqual(Limit{ .bytes = 4 << 30 }, resolve(null, 4));
    try std.testing.expectEqual(Limit{ .bytes = 3 << 29 }, resolve(null, 1.5));
}

test "prompt cache: --prompt-cache-gib 0 keeps no idle state resident" {
    try std.testing.expectEqual(Limit{ .bytes = 0 }, resolve(null, 0));
}

test "prompt cache: a set TF_DSV41_SESSION_RESIDENT_GIB wins over the flag and says so" {
    try std.testing.expectEqual(Limit{ .bytes = 1 << 30, .env_wins = true }, resolve(1 << 30, 4));
    try std.testing.expectEqual(Limit{ .bytes = 2 << 30, .env_wins = true }, resolve(2 << 30, 0));
    try std.testing.expectEqual(Limit{ .bytes = 1 << 30 }, resolve(1 << 30, null));
}

test "prompt cache: neither the flag nor the variable keeps today's unlimited default" {
    try std.testing.expectEqual(Limit{ .bytes = null }, resolve(null, null));
}

test "prompt cache: a follower's variable refuses only when set and not rank 0's limit, by name" {
    try std.testing.expect(!differs(null, 4 << 30));
    try std.testing.expect(!differs(null, null));
    try std.testing.expect(!differs(4 << 30, 4 << 30));
    try std.testing.expect(differs(2 << 30, 4 << 30));
    try std.testing.expect(differs(0, null));
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("sessions: TF_DSV41_SESSION_RESIDENT_GIB on rank 1 is 2147483648 bytes, but rank 0's resident limit is 4294967296 bytes; unset it on that rank or match rank 0", refusal(&buf, 1, 2 << 30, 4 << 30));
    try std.testing.expect(std.mem.indexOf(u8, refusal(&buf, 3, 0, null), "rank 0's resident limit is unlimited") != null);
}

/// Two ranks over loopback plan sockets, as session_protocol.zig's test: each rank's variable and rank 0's flag.
const Ranks = struct {
    bytes: [2]?u64 = .{ null, null },
    err: [2]?anyerror = .{ null, null },

    fn side(plan: *tp.planlink.PlanLink, env: ?u64, gib: ?f64, out: *?u64, err: *?anyerror) void {
        out.* = agree(plan, env, gib) catch |e| {
            err.* = e;
            return;
        };
    }

    fn run(env: [2]?u64, gib: ?f64) !Ranks {
        const listener = try tp.bootstrap.testListener();
        defer tp.sock.close(listener.fd);
        const deadline = tp.sock.nowNs() + timeout_ns;
        const follower_fd = try tp.sock.connect("localhost", listener.port, deadline);
        const root_fd = try tp.sock.accept(listener.fd, deadline);
        var root_fds: [tp.planlink.max_ranks]tp.sock.Fd = @splat(-1);
        var follower_fds = root_fds;
        root_fds[1] = root_fd;
        follower_fds[0] = follower_fd;
        var root = tp.planlink.PlanLink.init(0, 2, root_fds);
        defer root.close();
        var follower = tp.planlink.PlanLink.init(1, 2, follower_fds);
        defer follower.close();
        var r: Ranks = .{};
        // the follower takes no flag: `tf-dsv41 follow` has none
        const thread = try std.Thread.spawn(.{}, side, .{ &follower, env[1], null, &r.bytes[1], &r.err[1] });
        side(&root, env[0], gib, &r.bytes[0], &r.err[0]);
        thread.join();
        if (r.err[0] == null and r.err[1] == null) {
            // the exchange leaves the link on a frame boundary for the plans after it
            try root.send(&.{99});
            var next: [1]i64 = undefined;
            try std.testing.expectEqualSlices(i64, &.{99}, try follower.recv(&next));
        }
        return r;
    }
};

test "prompt cache: two ranks: rank 0's --prompt-cache-gib reaches the follower" {
    const t = std.testing;
    inline for (.{ 4, 0 }) |gib| {
        const r = try Ranks.run(.{ null, null }, gib);
        try t.expectEqual([2]?anyerror{ null, null }, r.err);
        try t.expectEqual([2]?u64{ gib << 30, gib << 30 }, r.bytes);
    }
    // a follower variable equal to rank 0's limit is accepted; rank 0's variable wins over its flag on both
    const same = try Ranks.run(.{ null, 4 << 30 }, 4);
    try t.expectEqual([2]?u64{ 4 << 30, 4 << 30 }, same.bytes);
    const env = try Ranks.run(.{ 1 << 30, 1 << 30 }, 4);
    try t.expectEqual([2]?anyerror{ null, null }, env.err);
    try t.expectEqual([2]?u64{ 1 << 30, 1 << 30 }, env.bytes);
}

test "prompt cache: two ranks: a follower's different TF_DSV41_SESSION_RESIDENT_GIB refuses on both" {
    const t = std.testing;
    const cases = [_]struct { env: [2]?u64, gib: ?f64 }{
        .{ .env = .{ null, 2 << 30 }, .gib = 4 },
        .{ .env = .{ null, 4 << 30 }, .gib = null },
        .{ .env = .{ 1 << 30, 4 << 30 }, .gib = 4 },
        .{ .env = .{ null, 0 }, .gib = 4 },
    };
    for (cases) |c| {
        const r = try Ranks.run(c.env, c.gib);
        try t.expectEqual([2]?anyerror{ error.SessionResidentMismatch, error.SessionResidentMismatch }, r.err);
    }
}

test "prompt cache: two ranks: no flag and no variable keeps today's unlimited default on both" {
    const r = try Ranks.run(.{ null, null }, null);
    try std.testing.expectEqual([2]?anyerror{ null, null }, r.err);
    try std.testing.expectEqual([2]?u64{ null, null }, r.bytes);
}

test "prompt cache: one rank resolves alone, with no plan socket" {
    var plan = tp.planlink.PlanLink.init(0, 1, @splat(-1));
    try std.testing.expectEqual(@as(?u64, 4 << 30), try agree(&plan, null, 4));
    try std.testing.expectEqual(@as(?u64, null), try agree(&plan, null, null));
    try std.testing.expectEqual(@as(?u64, 2 << 30), try agree(&plan, 2 << 30, 4));
}
