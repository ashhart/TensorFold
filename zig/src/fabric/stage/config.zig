//! A stage rank's ring settings from TF_STAGE_RANK, _WORLD, _PEERS, _SECRET_FILE, _TIMEOUT_S and _LAYOUT.
const std = @import("std");
const ring = @import("ring.zig");
const secret = @import("secret.zig");
const boot = @import("boot.zig");
const environment = @import("sys").environment;

/// The most ranks a ring takes.
pub const most_ranks = 64;

pub const Error = secret.Error || error{ MissingStageSetting, BadStageRank, BadStageWorld, BadStagePeers, BadStageTimeout, BadStageSecretFile, BadStageLayout };

/// Parsed settings; `peers` points into `slots` and into the TF_STAGE_PEERS text.
pub const Settings = struct {
    rank: u32,
    world: u32,
    timeout_ns: u64,
    key: secret.Key = undefined,
    slots: [most_ranks]ring.Peer = undefined,
    n: usize = 0,
    ranges: [most_ranks]boot.Range = undefined,

    pub fn peers(s: *const Settings) []const ring.Peer {
        return s.slots[0..s.n];
    }

    /// Every rank's layer range, in rank order.
    pub fn layout(s: *const Settings) []const boot.Range {
        return s.ranges[0..s.world];
    }

    /// The ring's options for these settings and the checkpoint's and layout's `fingerprint`.
    pub fn options(s: *const Settings, fingerprint: [32]u8) ring.Options {
        return .{ .rank = s.rank, .world = s.world, .peers = s.peers(), .fingerprint = fingerprint, .key = s.key, .timeout_ns = s.timeout_ns };
    }
};

/// One setting as its row in platform/environment.zig's table takes it: null when unset or blank, else `bad`.
fn setting(get: anytype, name: [:0]const u8, bad: Error) Error!?[]const u8 {
    const raw = get.lookup(name);
    environment.check(name, raw) catch return bad; // logged with the value and the accepted forms
    return if ((environment.parse(name, raw) catch return bad) == null) null else raw;
}

/// The settings named by `get` (an environment lookup), refused by name when one is missing or malformed.
pub fn parse(get: anytype, s: *Settings) Error!void {
    const rank = try setting(get, "TF_STAGE_RANK", error.BadStageRank) orelse return error.MissingStageSetting;
    const world = try setting(get, "TF_STAGE_WORLD", error.BadStageWorld) orelse return error.MissingStageSetting;
    const list = try setting(get, "TF_STAGE_PEERS", error.BadStagePeers) orelse return error.MissingStageSetting;
    const path = try setting(get, "TF_STAGE_SECRET_FILE", error.BadStageSecretFile) orelse return error.MissingStageSetting;
    const ranges = try setting(get, "TF_STAGE_LAYOUT", error.BadStageLayout) orelse return error.MissingStageSetting;
    const timeout = try setting(get, "TF_STAGE_TIMEOUT_S", error.BadStageTimeout);
    s.* = .{ .rank = 0, .world = 0, .timeout_ns = 60 * std.time.ns_per_s };
    s.world = std.fmt.parseInt(u32, world, 10) catch return error.BadStageWorld;
    s.rank = std.fmt.parseInt(u32, rank, 10) catch return error.BadStageRank;
    if (s.rank >= s.world) return error.BadStageRank;
    if (timeout) |v| s.timeout_ns = @as(u64, std.fmt.parseInt(u32, v, 10) catch return error.BadStageTimeout) * std.time.ns_per_s;
    var it = std.mem.splitScalar(u8, list, ',');
    while (it.next()) |entry| : (s.n += 1) {
        if (s.n == s.world) return error.BadStagePeers;
        const colon = std.mem.lastIndexOfScalar(u8, entry, ':').?;
        s.slots[s.n] = .{ .host = entry[0..colon], .port = std.fmt.parseInt(u16, entry[colon + 1 ..], 10) catch return error.BadStagePeers };
    }
    if (s.n != s.world) return error.BadStagePeers;
    var n: usize = 0;
    var rs = std.mem.splitScalar(u8, ranges, ',');
    while (rs.next()) |entry| : (n += 1) {
        if (n == s.world) return error.BadStageLayout;
        const colon = std.mem.indexOfScalar(u8, entry, ':').?;
        const first = std.fmt.parseInt(u32, entry[0..colon], 10) catch return error.BadStageLayout;
        s.ranges[n] = .{ .first = first, .end = std.fmt.parseInt(u32, entry[colon + 1 ..], 10) catch return error.BadStageLayout };
    }
    if (n != s.world) return error.BadStageLayout;
    var name: [4096]u8 = undefined;
    if (path.len >= name.len) return error.SecretUnreadable;
    @memcpy(name[0..path.len], path);
    name[path.len] = 0;
    s.key = try secret.fromFile(name[0..path.len :0]);
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
    secret_file: ?[]const u8 = "/nonexistent/stage-secret",
    layout: ?[]const u8 = "0:20,20:40,40:64",
    pub fn lookup(f: Fake, name: [:0]const u8) ?[]const u8 {
        if (std.mem.eql(u8, name, "TF_STAGE_RANK")) return f.rank;
        if (std.mem.eql(u8, name, "TF_STAGE_WORLD")) return f.world;
        if (std.mem.eql(u8, name, "TF_STAGE_PEERS")) return f.peers;
        if (std.mem.eql(u8, name, "TF_STAGE_TIMEOUT_S")) return f.timeout;
        if (std.mem.eql(u8, name, "TF_STAGE_SECRET_FILE")) return f.secret_file;
        if (std.mem.eql(u8, name, "TF_STAGE_LAYOUT")) return f.layout;
        return null;
    }
};

/// An owner-only secret file for one test, at `path`.
fn writeSecret(path: [:0]const u8) !void {
    _ = std.c.unlink(path);
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.SecretUnreadable;
    defer _ = std.c.close(fd);
    _ = std.c.write(fd, "a stage ring's test secret\n", 27);
}

test "stage config: a three-rank ring's settings parse, with its timeout and key" {
    var buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, "/tmp/tf-stage-config-{d}", .{std.c.getpid()}, 0);
    try writeSecret(path);
    defer _ = std.c.unlink(path);
    var s: Settings = undefined;
    try parse(Fake{ .timeout = "5", .secret_file = path }, &s);
    try std.testing.expectEqual(try secret.fromBytes("a stage ring's test secret"), s.key);
    try std.testing.expectEqual(@as(u32, 1), s.rank);
    try std.testing.expectEqual(@as(u32, 3), s.world);
    try std.testing.expectEqual(5 * std.time.ns_per_s, s.timeout_ns);
    try std.testing.expectEqualStrings("stage-b.example", s.peers()[1].host);
    try std.testing.expectEqual(@as(u16, 7102), s.peers()[2].port);
    try std.testing.expectEqualSlices(boot.Range, &.{ .{ .first = 0, .end = 20 }, .{ .first = 20, .end = 40 }, .{ .first = 40, .end = 64 } }, s.layout());
    try parse(Fake{ .secret_file = path }, &s);
    try std.testing.expectEqual(60 * std.time.ns_per_s, s.timeout_ns);
}

test "stage config: a missing or malformed setting is refused by name" {
    var s: Settings = undefined;
    try std.testing.expectError(error.MissingStageSetting, parse(Fake{ .peers = null }, &s));
    try std.testing.expectError(error.MissingStageSetting, parse(Fake{ .secret_file = null }, &s));
    try std.testing.expectError(error.MissingStageSetting, parse(Fake{ .layout = " " }, &s));
    try std.testing.expectError(error.SecretUnreadable, parse(Fake{}, &s));
    try std.testing.expectError(error.BadStageWorld, parse(Fake{ .world = "1" }, &s));
    try std.testing.expectError(error.BadStageWorld, parse(Fake{ .world = "65" }, &s));
    try std.testing.expectError(error.BadStageRank, parse(Fake{ .rank = "3" }, &s));
    try std.testing.expectError(error.BadStageRank, parse(Fake{ .rank = "-1" }, &s));
    try std.testing.expectError(error.BadStageTimeout, parse(Fake{ .timeout = "0" }, &s));
    try std.testing.expectError(error.BadStagePeers, parse(Fake{ .peers = "a:1,b:2" }, &s));
    try std.testing.expectError(error.BadStagePeers, parse(Fake{ .peers = "a:1,b:2,c:3,d:4" }, &s));
    try std.testing.expectError(error.BadStagePeers, parse(Fake{ .peers = "a:1,b:0,c:3" }, &s));
    try std.testing.expectError(error.BadStagePeers, parse(Fake{ .peers = "a:1,:2,c:3" }, &s));
    try std.testing.expectError(error.BadStagePeers, parse(Fake{ .peers = "a:1,b,c:3" }, &s));
    try std.testing.expectError(error.BadStageSecretFile, parse(Fake{ .secret_file = "stage\nsecret" }, &s));
    try std.testing.expectError(error.BadStageLayout, parse(Fake{ .layout = "0:32,32:64" }, &s));
    try std.testing.expectError(error.BadStageLayout, parse(Fake{ .layout = "0:20,20:40,40:64,64:70" }, &s));
    try std.testing.expectError(error.BadStageLayout, parse(Fake{ .layout = "0:20,20:x,40:64" }, &s));
}

test "stage config: every TF_STAGE_ setting it reads is a row of the environment table, refused by its own name" {
    const Want = struct { name: []const u8, err: Error, fake: Fake };
    const all = [_]Want{
        .{ .name = "TF_STAGE_RANK", .err = error.BadStageRank, .fake = .{ .rank = "typo" } },
        .{ .name = "TF_STAGE_WORLD", .err = error.BadStageWorld, .fake = .{ .world = "typo" } },
        .{ .name = "TF_STAGE_PEERS", .err = error.BadStagePeers, .fake = .{ .peers = "typo" } },
        .{ .name = "TF_STAGE_SECRET_FILE", .err = error.BadStageSecretFile, .fake = .{ .secret_file = "key\nfile" } },
        .{ .name = "TF_STAGE_TIMEOUT_S", .err = error.BadStageTimeout, .fake = .{ .timeout = "typo" } },
        .{ .name = "TF_STAGE_LAYOUT", .err = error.BadStageLayout, .fake = .{ .layout = "typo" } },
    };
    var n: usize = 0;
    for (environment.rules) |r| if (r.scope == .stage) {
        n += 1;
        const w = for (all) |w| {
            if (std.mem.eql(u8, w.name, r.name)) break w;
        } else return error.TestUnexpectedResult;
        try std.testing.expectError(error.InvalidEnvironment, environment.parse(r.name, w.fake.lookup(r.name).?));
        var s: Settings = undefined;
        try std.testing.expectError(w.err, parse(w.fake, &s));
    };
    try std.testing.expectEqual(all.len, n);
}
