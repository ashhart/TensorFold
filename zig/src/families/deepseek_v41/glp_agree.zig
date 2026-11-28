//! Agree on the full loaded GLPv3 additive policy before weights; every TP rank refuses a mismatch.
const std = @import("std");
const tp = @import("tp");
const glp_plan = @import("glp_plan.zig");
const quiet = @import("builtin").is_test;
const timeout_ns = 5 * std.time.ns_per_s;
const Words = [8]i64;

fn send(plan: *tp.planlink.PlanLink, peer: usize, words: []const i64, deadline: u64) !void {
    try tp.sock.sendBefore(plan.fds[peer], std.mem.sliceAsBytes(words), deadline);
}
fn recv(plan: *tp.planlink.PlanLink, peer: usize, words: []i64, deadline: u64) !void {
    try tp.sock.recvExact(plan.fds[peer], std.mem.sliceAsBytes(words), deadline);
}
fn refuse(rank: usize) void {
    if (!quiet) std.log.scoped(.dsv41).err("glp: rank {d} differs from rank 0; every rank needs the same file, site and strength, including inactive loads", .{rank});
}

pub fn agree(plan: *tp.planlink.PlanLink, mine: Words) !void {
    if (plan.world <= 1) return;
    const deadline = tp.sock.nowNs() + timeout_ns;
    if (plan.rank != 0) {
        var leader: Words = undefined;
        try recv(plan, 0, &leader, deadline);
        try send(plan, 0, &mine, deadline);
        var verdict: [1]i64 = undefined;
        try recv(plan, 0, &verdict, deadline);
        const same = std.mem.eql(i64, &leader, &mine);
        if (!same) refuse(plan.rank);
        if (!same or verdict[0] != 1) return error.GlpMismatch;
        return;
    }
    var ok = true;
    for (1..plan.world) |rank| {
        try send(plan, rank, &mine, deadline);
        var peer: Words = undefined;
        try recv(plan, rank, &peer, deadline);
        if (!std.mem.eql(i64, &peer, &mine)) {
            refuse(rank);
            ok = false;
        }
    }
    const verdict = [_]i64{@intFromBool(ok)};
    for (1..plan.world) |rank| try send(plan, rank, &verdict, deadline);
    if (!ok) return error.GlpMismatch;
}

const Ranks = struct {
    err: [2]?anyerror = .{ null, null },
    fn side(plan: *tp.planlink.PlanLink, own: Words, err: *?anyerror) void {
        agree(plan, own) catch |e| {
            err.* = e;
        };
    }
    fn run(own: [2]Words) !Ranks {
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
        const thread = try std.Thread.spawn(.{}, side, .{ &follower, own[1], &r.err[1] });
        side(&root, own[0], &r.err[0]);
        thread.join();
        return r;
    }
};

const Source = struct {
    alpha: f32,
    vector: struct { spec_version: u32 = 3, file_sha256: [32]u8, layers: []const u32 },
};
fn policy(alpha: f32, tag: u8) Words {
    const source: ?Source = .{ .alpha = alpha, .vector = .{ .file_sha256 = @splat(tag), .layers = &.{1} } };
    return glp_plan.policyWords(source);
}

test "two real plan links agree on absence and on the same loaded additive policy" {
    const none = glp_plan.policyWords(@as(?Source, null));
    try std.testing.expectEqual([2]?anyerror{ null, null }, (try Ranks.run(.{ none, none })).err);
    const a = policy(0.5, 1);
    try std.testing.expectEqual([2]?anyerror{ null, null }, (try Ranks.run(.{ a, a })).err);
}

test "different files, strength or presence refuse on both ranks" {
    const a = policy(0.5, 1);
    const none = glp_plan.policyWords(@as(?Source, null));
    const both = [2]?anyerror{ error.GlpMismatch, error.GlpMismatch };
    for ([_]Words{ policy(0.5, 2), policy(0.25, 1), none }) |other| {
        try std.testing.expectEqual(both, (try Ranks.run(.{ a, other })).err);
        try std.testing.expectEqual(both, (try Ranks.run(.{ other, a })).err);
    }
}

test "zero strength remains a loaded policy, with both signs of zero equivalent" {
    const zero = policy(0, 1);
    const none = glp_plan.policyWords(@as(?Source, null));
    const both = [2]?anyerror{ error.GlpMismatch, error.GlpMismatch };
    try std.testing.expectEqual(both, (try Ranks.run(.{ zero, none })).err);
    try std.testing.expectEqual(both, (try Ranks.run(.{ none, zero })).err);
    try std.testing.expectEqual(both, (try Ranks.run(.{ zero, policy(0, 2) })).err);
    try std.testing.expectEqual([2]?anyerror{ null, null }, (try Ranks.run(.{ zero, policy(-0.0, 1) })).err);
}
