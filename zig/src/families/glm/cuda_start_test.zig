//! The actual startup boundary with two recording liveness peers, without sockets, CUDA or checkpoints.
const std = @import("std");
const startup = @import("cuda_start.zig");

const Settings = struct { kv8: bool, pages: bool = false, marker: u32 = 37 };
const Place = struct { rank: usize, world: usize = 2 };
const World = struct {
    connected: [2]bool = @splat(false),
    waiting: [2]bool = @splat(false),
    refused: [2]?[]const u8 = @splat(null),
    constructed: [2]bool = @splat(false),
    settings: [2]?Settings = @splat(null),
    starts: usize = 0,
    notifications: usize = 0,
    fail_start: bool = false,
    fail_engine: bool = false,
};
const Watch = struct {
    world: *World,
    rank: usize,

    pub fn start(w: Watch, _: std.Io, rank: usize, world: usize) !void {
        try std.testing.expectEqual(w.rank, rank);
        try std.testing.expectEqual(@as(usize, 2), world);
        w.world.starts += 1;
        if (w.world.fail_start) return error.InjectedConnectFailure;
        w.world.connected[rank] = true;
    }

    pub fn fail(w: Watch, name: []const u8) void {
        w.world.notifications += 1;
        const peer = 1 - w.rank;
        if (w.world.connected[w.rank] and w.world.connected[peer]) {
            w.world.refused[peer] = name;
            w.world.waiting[peer] = false;
        }
    }
};
const Args = struct { world: *World, rank: usize };

fn construct(a: Args, settings: Settings) anyerror!void {
    a.world.constructed[a.rank] = true;
    a.world.settings[a.rank] = settings;
    if (a.world.fail_engine) return error.InjectedEngineFailure;
    a.world.waiting[a.rank] = true;
}

fn run(w: *World, rank: usize, world: usize, text: ?[]const u8, kv8: bool) !void {
    return startup.open(std.testing.io, Place{ .rank = rank, .world = world }, text, Settings{ .kv8 = kv8 }, Watch{ .world = w, .rank = rank }, Args{ .world = w, .rank = rank }, construct);
}

fn refusal(rank: usize) !void {
    for ([_][]const u8{ "on", "all", "1" }) |mode| {
        var w: World = .{};
        const peer = 1 - rank;
        try run(&w, peer, 2, "auto", true);
        try std.testing.expect(w.waiting[peer]);
        try std.testing.expectError(error.Fp8OnPages, run(&w, rank, 2, mode, true));
        try std.testing.expectEqualStrings("Fp8OnPages", w.refused[peer] orelse "waiting for missing rank");
        try std.testing.expect(!w.waiting[peer]);
        try std.testing.expect(w.connected[rank] and w.connected[peer]);
        try std.testing.expect(!w.constructed[rank]);
        try std.testing.expectEqual(@as(usize, 2), w.starts);
        try std.testing.expectEqual(@as(usize, 1), w.notifications);
    }
}

test "GLM startup: refusing rank zero promptly tells the waiting follower Fp8OnPages" {
    try refusal(0);
}

test "GLM startup: refusing follower promptly tells waiting rank zero Fp8OnPages" {
    try refusal(1);
}

test "GLM startup: valid modes reach construction with every other setting intact" {
    for ([_]bool{ false, true }) |kv8| for ([_][]const u8{ "auto", "off", "0", "on", "all" }) |mode| {
        if (kv8 and (std.mem.eql(u8, mode, "on") or std.mem.eql(u8, mode, "all"))) continue;
        var w: World = .{};
        try run(&w, 0, 2, mode, kv8);
        try std.testing.expect(w.connected[0]);
        try std.testing.expect(w.constructed[0]);
        try std.testing.expectEqual(try startup.paging(mode, 2, kv8), w.settings[0].?.pages);
        try std.testing.expectEqual(kv8, w.settings[0].?.kv8);
        try std.testing.expectEqual(@as(u32, 37), w.settings[0].?.marker);
        try std.testing.expectEqual(@as(usize, 0), w.notifications);
    };
    var lone: World = .{};
    try run(&lone, 0, 1, "all", false);
    try std.testing.expect(lone.settings[0].?.pages);
    try std.testing.expectEqual(@as(usize, 0), lone.starts);
}

test "GLM startup: construction errors notify peers and preserve their actual name" {
    for (0..2) |rank| {
        var w: World = .{};
        const peer = 1 - rank;
        try run(&w, peer, 2, "off", false);
        w.fail_engine = true;
        try std.testing.expectError(error.InjectedEngineFailure, run(&w, rank, 2, "on", false));
        try std.testing.expectEqualStrings("InjectedEngineFailure", w.refused[peer] orelse "waiting");
        try std.testing.expect(w.settings[rank].?.pages);
        try std.testing.expect(!w.waiting[peer]);
    }
}

test "GLM startup: failed liveness connection never resolves settings or constructs an engine" {
    var w: World = .{ .fail_start = true };
    try std.testing.expectError(error.InjectedConnectFailure, run(&w, 0, 2, "on", true));
    try std.testing.expect(!w.constructed[0]);
    try std.testing.expectEqual(@as(usize, 0), w.notifications);
}
