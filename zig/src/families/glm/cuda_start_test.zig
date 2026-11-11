//! The actual startup boundary with two recording liveness peers, without sockets, CUDA or checkpoints.
const std = @import("std");
const startup = @import("cuda_start.zig");

fn recordReport(args: Args, selection: startup.Report) void {
    const w = args.world;
    const line = std.fmt.bufPrint(&w.report, "glm kv selection: {f}", .{selection}) catch @panic("unbounded startup report");
    w.report_len = line.len;
    w.reports += 1;
}

const Settings = struct { kv8: bool, pages: bool = false, marker: u32 = 37, kv_split: bool = false, kv_host: u8 = 0 };
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
    expected_world: usize = 2,
    reports: usize = 0,
    report: [256]u8 = undefined,
    report_len: usize = 0,
};
const Watch = struct {
    world: *World,
    rank: usize,

    pub fn start(w: Watch, _: std.Io, rank: usize, world: usize) !void {
        try std.testing.expectEqual(w.rank, rank);
        try std.testing.expectEqual(w.world.expected_world, world);
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
    try std.testing.expect(a.world.reports > 0);
    a.world.constructed[a.rank] = true;
    a.world.settings[a.rank] = settings;
    if (a.world.fail_engine) return error.InjectedEngineFailure;
    a.world.waiting[a.rank] = true;
}

fn run(w: *World, rank: usize, world: usize, text: ?[]const u8, kv8: bool) !void {
    return startup.open(std.testing.io, Place{ .rank = rank, .world = world }, text, Settings{ .kv8 = kv8 }, Watch{ .world = w, .rank = rank }, Args{ .world = w, .rank = rank }, construct, recordReport);
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
        try std.testing.expectEqual(@as(usize, 1), w.reports);
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
    try std.testing.expectEqual(@as(usize, 0), w.reports);
}

test "GLM startup: legacy paging matrix preserves settings and reports each rank before construction" {
    const Case = struct { raw: ?[]const u8, single: bool, multi: bool, fp8_refuses: bool, mode: []const u8 };
    const cases = [_]Case{
        .{ .raw = null, .single = false, .multi = true, .fp8_refuses = false, .mode = "auto" },
        .{ .raw = "auto", .single = false, .multi = true, .fp8_refuses = false, .mode = "auto" },
        .{ .raw = "off", .single = false, .multi = false, .fp8_refuses = false, .mode = "off" },
        .{ .raw = "0", .single = false, .multi = false, .fp8_refuses = false, .mode = "off" },
        .{ .raw = "on", .single = true, .multi = true, .fp8_refuses = true, .mode = "on" },
        .{ .raw = "1", .single = true, .multi = true, .fp8_refuses = true, .mode = "on" },
        .{ .raw = "all", .single = true, .multi = true, .fp8_refuses = true, .mode = "on" },
        .{ .raw = "", .single = true, .multi = true, .fp8_refuses = true, .mode = "on" },
        .{ .raw = "false", .single = true, .multi = true, .fp8_refuses = true, .mode = "on" },
        .{ .raw = "AUTO", .single = true, .multi = true, .fp8_refuses = true, .mode = "on" },
        .{ .raw = "00", .single = true, .multi = true, .fp8_refuses = true, .mode = "on" },
        .{ .raw = " 0 ", .single = true, .multi = true, .fp8_refuses = true, .mode = "on" },
        .{ .raw = "off\n\x1b[31m", .single = true, .multi = true, .fp8_refuses = true, .mode = "on" },
    };
    for (cases) |c| for (1..17) |world| for ([_]bool{ false, true }) |kv8| {
        const last_rank = @min(world, 2);
        for (0..last_rank) |rank| {
            var w: World = .{ .expected_world = world };
            if (kv8 and c.fp8_refuses) {
                try std.testing.expectError(error.Fp8OnPages, run(&w, rank, world, c.raw, kv8));
                try std.testing.expect(!w.constructed[rank]);
                try std.testing.expectEqual(@as(usize, 0), w.reports);
                try std.testing.expectEqual(@as(usize, 1), w.notifications);
                continue;
            }
            const pages = !kv8 and (if (world == 1) c.single else c.multi);
            try run(&w, rank, world, c.raw, kv8);
            try std.testing.expectEqualDeep(Settings{ .kv8 = kv8, .pages = pages }, w.settings[rank].?);
            try std.testing.expectEqual(@as(usize, 1), w.reports);
            var buffer: [256]u8 = undefined;
            const expected = try std.fmt.bufPrint(&buffer, "glm kv selection: {s}; TF_KV_PAGES={s} ({s}); rank {d} of {d}; cache {s}", .{
                if (pages) "pool pages" else "window per stream", c.mode,
                if (c.raw == null) "default" else "env",          rank,
                world,                                            if (kv8) "fp8" else "bf16",
            });
            try std.testing.expectEqualStrings(expected, w.report[0..w.report_len]);
        }
    };
}

test "GLM startup: reports use the passed selection and never echo arbitrary environment bytes" {
    const raw: [4096]u8 = @splat('\n');
    var buffer: [256]u8 = undefined;
    const report: startup.Report = .{ .text = &raw, .rank = 15, .world = 16, .kv8 = false, .pages = false };
    const line = try std.fmt.bufPrint(&buffer, "{f}", .{report});
    try std.testing.expectEqualStrings("window per stream; TF_KV_PAGES=on (env); rank 15 of 16; cache bf16", line);
}

fn runSplit(w: *World, rank: usize, world: usize, text: ?[]const u8, kv8: bool, host: u8) !void {
    return startup.open(std.testing.io, Place{ .rank = rank, .world = world }, text, Settings{ .kv8 = kv8, .kv_split = true, .kv_host = host }, Watch{ .world = w, .rank = rank }, Args{ .world = w, .rank = rank }, construct, recordReport);
}

test "GLM startup: TF_GLM_KV_SPLIT is refused by name at one rank, off pages or host pages, the peer told, before the engine" {
    const Case = struct { world: usize, text: ?[]const u8, kv8: bool = false, host: u8 = 0, err: anyerror };
    for ([_]Case{
        .{ .world = 1, .text = null, .err = error.KvSplitNeedsRanks }, // auto pages are off at one rank: ranks named first
        .{ .world = 1, .text = "on", .err = error.KvSplitNeedsRanks },
        .{ .world = 2, .text = "off", .err = error.KvSplitNeedsPages },
        .{ .world = 2, .text = null, .kv8 = true, .err = error.KvSplitNeedsPages }, // fp8 keeps windows under auto
        .{ .world = 2, .text = "on", .host = 1, .err = error.KvSplitOnHost },
    }) |c| for (0..c.world) |rank| {
        var w: World = .{ .expected_world = c.world };
        if (c.world > 1) try run(&w, 1 - rank, 2, c.text, c.kv8);
        try std.testing.expectError(c.err, runSplit(&w, rank, c.world, c.text, c.kv8, c.host));
        try std.testing.expect(!w.constructed[rank]);
        if (c.world > 1) try std.testing.expectEqualStrings(@errorName(c.err), w.refused[1 - rank] orelse "waiting");
    };
    for ([_]?[]const u8{ null, "auto", "on" }) |text| {
        var w: World = .{};
        try runSplit(&w, 0, 2, text, false, 0);
        try std.testing.expect(w.settings[0].?.kv_split and w.settings[0].?.pages);
    }
    var off: World = .{};
    try run(&off, 0, 2, null, false);
    try std.testing.expect(!off.settings[0].?.kv_split);
}

test "GLM startup: TF_GLM_KV_SPLIT with TF_KV_SHARE is refused by name, either alone is not" {
    for ([_]usize{ 1, 4, 64 }) |share| try std.testing.expectError(error.KvSplitShare, startup.sharing(true, share));
    try startup.sharing(true, 0);
    for ([_]usize{ 0, 1, 64 }) |share| try startup.sharing(false, share);
}

test "GLM startup: TF_GLM_KV_SPLIT with --learn-conversations is refused by name, either alone is not" {
    try std.testing.expectError(error.KvSplitLearn, startup.learning(true, true));
    try startup.learning(true, false);
    for ([_]bool{ false, true }) |learn| try startup.learning(false, learn);
}

test "GLM startup: every TF_GLM_KV_SPLIT refusal is explained, and another error is left to its family" {
    for ([_]anyerror{ error.KvSplitNeedsRanks, error.KvSplitNeedsPages, error.KvSplitOnHost, error.KvSplitShare, error.KvSplitLearn, error.KvSplitAway, error.KvSplitCarry, error.ClassTooSmall, error.ResidueMismatch, error.BadClasses }) |e| try std.testing.expect(startup.explain(e) != null);
    try std.testing.expect(startup.explain(error.Fp8OnPages) == null);
}
