//! A stage rank's ring settings from TF_STAGE_RANK, TF_STAGE_WORLD, TF_STAGE_PEERS and TF_STAGE_TIMEOUT_S.
const std = @import("std");
const ring = @import("ring.zig");

/// The most ranks a ring takes.
pub const most_ranks = 64;

pub const Error = error{ MissingStageSetting, BadStageRank, BadStageWorld, BadStagePeers, BadStageTimeout };

/// Parsed settings; `peers` points into `slots` and into the TF_STAGE_PEERS text.
pub const Settings = struct {
    rank: u32,
    world: u32,
    timeout_ns: u64,
    slots: [most_ranks]ring.Peer = undefined,
    n: usize = 0,

    pub fn peers(s: *const Settings) []const ring.Peer {
        return s.slots[0..s.n];
    }

    /// The ring's options for these settings and the checkpoint's and layout's `fingerprint`.
    pub fn options(s: *const Settings, fingerprint: [32]u8) ring.Options {
        return .{ .rank = s.rank, .world = s.world, .peers = s.peers(), .fingerprint = fingerprint, .timeout_ns = s.timeout_ns };
    }
};

/// The settings named by `get` (an environment lookup), refused by name when one is missing or malformed.
pub fn parse(get: anytype, s: *Settings) Error!void {
    const rank = get.lookup("TF_STAGE_RANK") orelse return error.MissingStageSetting;
    const world = get.lookup("TF_STAGE_WORLD") orelse return error.MissingStageSetting;
    const list = get.lookup("TF_STAGE_PEERS") orelse return error.MissingStageSetting;
    s.* = .{ .rank = 0, .world = 0, .timeout_ns = 60 * std.time.ns_per_s };
    s.world = std.fmt.parseInt(u32, world, 10) catch return error.BadStageWorld;
    if (s.world < 2 or s.world > most_ranks) return error.BadStageWorld;
    s.rank = std.fmt.parseInt(u32, rank, 10) catch return error.BadStageRank;
    if (s.rank >= s.world) return error.BadStageRank;
    if (get.lookup("TF_STAGE_TIMEOUT_S")) |v| {
        const secs = std.fmt.parseInt(u32, v, 10) catch return error.BadStageTimeout;
        if (secs == 0) return error.BadStageTimeout;
        s.timeout_ns = @as(u64, secs) * std.time.ns_per_s;
    }
    var it = std.mem.splitScalar(u8, list, ',');
    while (it.next()) |entry| {
        if (s.n == most_ranks) return error.BadStagePeers;
        const colon = std.mem.lastIndexOfScalar(u8, entry, ':') orelse return error.BadStagePeers;
        const host = entry[0..colon];
        const port = std.fmt.parseInt(u16, entry[colon + 1 ..], 10) catch return error.BadStagePeers;
        if (host.len == 0 or port == 0) return error.BadStagePeers;
        s.slots[s.n] = .{ .host = host, .port = port };
        s.n += 1;
    }
    if (s.n != s.world) return error.BadStagePeers;
}

/// The process environment, for `parse`.
pub const Env = struct {
    pub fn lookup(_: Env, name: [:0]const u8) ?[]const u8 {
        const v = std.c.getenv(name) orelse return null;
        return std.mem.span(v);
    }
};

const Fake = struct {
    rank: ?[]const u8 = "1",
    world: ?[]const u8 = "3",
    peers: ?[]const u8 = "192.0.2.1:7100,stage-b.example:7101,127.0.0.1:7102",
    timeout: ?[]const u8 = null,
    pub fn lookup(f: Fake, name: [:0]const u8) ?[]const u8 {
        if (std.mem.eql(u8, name, "TF_STAGE_RANK")) return f.rank;
        if (std.mem.eql(u8, name, "TF_STAGE_WORLD")) return f.world;
        if (std.mem.eql(u8, name, "TF_STAGE_PEERS")) return f.peers;
        if (std.mem.eql(u8, name, "TF_STAGE_TIMEOUT_S")) return f.timeout;
        return null;
    }
};

test "stage config: a three-rank ring's settings parse, with its timeout" {
    var s: Settings = undefined;
    try parse(Fake{ .timeout = "5" }, &s);
    try std.testing.expectEqual(@as(u32, 1), s.rank);
    try std.testing.expectEqual(@as(u32, 3), s.world);
    try std.testing.expectEqual(5 * std.time.ns_per_s, s.timeout_ns);
    try std.testing.expectEqualStrings("stage-b.example", s.peers()[1].host);
    try std.testing.expectEqual(@as(u16, 7102), s.peers()[2].port);
    try parse(Fake{}, &s);
    try std.testing.expectEqual(60 * std.time.ns_per_s, s.timeout_ns);
}

test "stage config: a missing or malformed setting is refused by name" {
    var s: Settings = undefined;
    try std.testing.expectError(error.MissingStageSetting, parse(Fake{ .peers = null }, &s));
    try std.testing.expectError(error.BadStageWorld, parse(Fake{ .world = "1" }, &s));
    try std.testing.expectError(error.BadStageWorld, parse(Fake{ .world = "65" }, &s));
    try std.testing.expectError(error.BadStageRank, parse(Fake{ .rank = "3" }, &s));
    try std.testing.expectError(error.BadStageRank, parse(Fake{ .rank = "-1" }, &s));
    try std.testing.expectError(error.BadStageTimeout, parse(Fake{ .timeout = "0" }, &s));
    try std.testing.expectError(error.BadStagePeers, parse(Fake{ .peers = "a:1,b:2" }, &s));
    try std.testing.expectError(error.BadStagePeers, parse(Fake{ .peers = "a:1,b:0,c:3" }, &s));
    try std.testing.expectError(error.BadStagePeers, parse(Fake{ .peers = "a:1,:2,c:3" }, &s));
    try std.testing.expectError(error.BadStagePeers, parse(Fake{ .peers = "a:1,b,c:3" }, &s));
}
