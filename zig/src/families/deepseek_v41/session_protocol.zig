//! Host control-channel preflight: every rank accepts the same session protocol before model boot plans.
const std = @import("std");
const tp = @import("tp");

pub const identity = [2]i64{ 0x4453563431534553, 3 };
const timeout_ns = 5 * std.time.ns_per_s;

pub fn preflight(plan: *tp.planlink.PlanLink) !void {
    try agree(plan, identity);
}

/// Rank 0's CLI value wins over follower defaults; bounded socket I/O finishes before any checkpoint boot.
pub fn cacheBudget(plan: *tp.planlink.PlanLink, mine: u64) !u64 {
    if (plan.world == 0 or plan.rank >= plan.world or plan.world > tp.planlink.max_ranks)
        return error.BadSessionProtocolRank;
    if (mine > std.math.maxInt(i64)) return error.BadCacheBudget;
    if (plan.world == 1) return mine;
    const deadline = tp.sock.nowNs() + timeout_ns;
    var words = [_]i64{@intCast(mine)};
    if (plan.rank == 0) {
        for (1..plan.world) |rank| try send(plan, rank, &words, deadline);
    } else try recv(plan, 0, &words, deadline);
    if (words[0] < 0) return error.BadCacheBudget;
    return @intCast(words[0]);
}

fn send(plan: *tp.planlink.PlanLink, peer: usize, words: []const i64, deadline: u64) !void {
    try tp.sock.sendBefore(plan.fds[peer], std.mem.sliceAsBytes(words), deadline);
}

fn recv(plan: *tp.planlink.PlanLink, peer: usize, words: []i64, deadline: u64) !void {
    try tp.sock.recvExact(plan.fds[peer], std.mem.sliceAsBytes(words), deadline);
}

fn agree(plan: *tp.planlink.PlanLink, mine: [2]i64) !void {
    if (plan.world == 0 or plan.rank >= plan.world or plan.world > tp.planlink.max_ranks)
        return error.BadSessionProtocolRank;
    if (plan.world == 1) return;
    const deadline = tp.sock.nowNs() + timeout_ns;
    var peer: [2]i64 = undefined;
    if (plan.rank == 0) {
        var ok = true;
        for (1..plan.world) |rank| {
            try send(plan, rank, &mine, deadline);
            try recv(plan, rank, &peer, deadline);
            ok = ok and std.mem.eql(i64, &mine, &peer);
        }
        const verdict = [_]i64{@intFromBool(ok)};
        for (1..plan.world) |rank| try send(plan, rank, &verdict, deadline);
        if (!ok) return error.SessionProtocolMismatch;
    } else {
        try recv(plan, 0, &peer, deadline);
        try send(plan, 0, &mine, deadline);
        var verdict: [1]i64 = undefined;
        try recv(plan, 0, &verdict, deadline);
        if (verdict[0] != 1 or !std.mem.eql(i64, &mine, &peer)) return error.SessionProtocolMismatch;
    }
}

fn run(plan: *tp.planlink.PlanLink, mine: [2]i64, result: *?anyerror) void {
    agree(plan, mine) catch |err| {
        result.* = err;
    };
}

test "session protocol: real host plan sockets accept identical ranks and both refuse either changed identity" {
    const t = std.testing;
    for (0..3) |scenario| {
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
        var root_identity = identity;
        var follower_identity = identity;
        if (scenario == 1) root_identity[1] += 1;
        if (scenario == 2) follower_identity[0] ^= 1;
        var result: [2]?anyerror = .{ null, null };
        const thread = try std.Thread.spawn(.{}, run, .{ &follower, follower_identity, &result[1] });
        run(&root, root_identity, &result[0]);
        thread.join();
        const expected: ?anyerror = if (scenario == 0) null else error.SessionProtocolMismatch;
        for (result) |err| try t.expectEqual(expected, err);
        if (scenario == 0) {
            try root.send(&.{99});
            var next: [1]i64 = undefined;
            try t.expectEqualSlices(i64, &.{99}, try follower.recv(&next));
        }
    }
}

test "session protocol: solo uses no sockets and invalid ranks fail before control I/O" {
    var plan = tp.planlink.PlanLink.init(0, 1, @splat(-1));
    try preflight(&plan);
    plan.world = 0;
    try std.testing.expectError(error.BadSessionProtocolRank, preflight(&plan));
    plan.world = 2;
    plan.rank = 2;
    try std.testing.expectError(error.BadSessionProtocolRank, preflight(&plan));
}

fn budgetRun(plan: *tp.planlink.PlanLink, mine: u64, result: *u64) void {
    preflight(plan) catch unreachable;
    result.* = cacheBudget(plan, mine) catch unreachable;
}

test "session protocol: root cache budget overrides followers before the next plan, including zero" {
    const t = std.testing;
    for ([_]u64{ 0, 1 << 28, 4 << 30, 16 << 30 }) |cap| {
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
        var got: [2]u64 = undefined;
        const thread = try std.Thread.spawn(.{}, budgetRun, .{ &follower, 99, &got[1] });
        budgetRun(&root, cap, &got[0]);
        thread.join();
        try t.expectEqual([2]u64{ cap, cap }, got);
        try root.send(&.{ 73, 9, 11 });
        var next: [3]i64 = undefined;
        try t.expectEqualSlices(i64, &.{ 73, 9, 11 }, try follower.recv(&next));
    }
}
