//! CPU records through the production prompt dispatcher; no HIP driver is loaded.
const std = @import("std");
const lanes = @import("lanes");
const source = @import("hip_lanes.zig");
const Engine = @import("engine.zig").Engine;
const prefix = @import("prefix.zig");
const T = std.testing;

const Flag = struct {
    on: bool = false,
    fn check(ptr: *anyopaque) bool {
        const flag: *Flag = @ptrCast(@alignCast(ptr));
        return flag.on;
    }
};

const Record = struct {
    flag: *Flag,
    world: usize = 1,
    cancel_inside: bool = false,
    advance_calls: usize = 0,
    prefill_calls: usize = 0,
    forwarded: ?Engine.Cancel = null,
    rows: usize = 0,
    o: struct { policy: struct { fill_rows: usize = 512 } = .{} } = .{},
    drawer: struct { vocab: usize = 8 } = .{},

    fn capture(e: *Record, rows: usize, cancel: ?Engine.Cancel) !void {
        e.forwarded = cancel;
        e.rows = rows;
        if (e.cancel_inside) e.flag.on = true;
        if (prefix.pollsLayers(e.world, rows)) {
            if (cancel) |c| if (c.check(c.ctx)) return error.Cancelled;
        }
    }

    pub fn advance(e: *Record, caches: anytype, prompt: []const u32, at: usize, to: usize, c: ?Engine.Cancel) !void {
        _ = caches;
        _ = prompt;
        e.advance_calls += 1;
        try e.capture(to - at, c);
    }

    pub fn prefill(e: *Record, caches: anytype, prompt: []const u32, at: usize, hidden: anytype, options: anytype, c: ?Engine.Cancel) !u32 {
        _ = caches;
        _ = hidden;
        _ = options;
        e.prefill_calls += 1;
        try e.capture(prompt.len - at, c);
        return 7;
    }
};

const Lookup = struct {
    lane: *source.Lane,
    pub fn get(l: Lookup, s: *lanes.Stream) ?*source.Lane {
        _ = s;
        return l.lane;
    }
};

const Prefix = struct {
    kept: usize = 0,
    pub fn keep(p: *Prefix, owner: anytype, prompt: []const u32, to: usize) !void {
        _ = owner;
        _ = prompt;
        _ = to;
        p.kept += 1;
    }
};

const Host = struct {
    gpa: std.mem.Allocator = T.allocator,
    e: *Record,
    lanes: Lookup,
    prefix: *Prefix,
    link: ?u32 = null,
    taken: usize = 0,
    pub fn begin(h: *Host, s: *lanes.Stream) !*source.Lane {
        _ = h;
        _ = s;
        return error.UnexpectedBegin;
    }
    pub fn decoding(h: *Host) bool {
        _ = h;
        return true;
    }
    pub fn send(h: *Host, words: []const u32) !void {
        _ = h;
        _ = words;
        return error.UnexpectedSend;
    }
    pub fn take(h: *Host, token: u32) u64 {
        std.debug.assert(token == 7);
        h.taken += 1;
        return 0;
    }
};

fn run(final: bool, rows: usize, world: usize, hook: bool, cancel_inside: bool) !void {
    var tokens: [2048]u32 = @splat(1);
    var flag: Flag = .{};
    var s = try lanes.Stream.init(T.allocator, .{
        .id = "prompt",
        .prompt = tokens[0..if (final) rows else rows + 1],
        .max_new = 8,
        .cancel_check = if (hook) .{ .ptr = &flag, .check = Flag.check } else null,
    });
    defer s.deinit(T.allocator);
    var lane: source.Lane = undefined;
    lane.fill = .{ .at = 0, .stops = @splat(@intCast(rows)), .count = if (final) 0 else 1, .next = 0 };
    lane.first_row = null;
    lane.id = 0;
    var e: Record = .{ .flag = &flag, .world = world, .cancel_inside = cancel_inside };
    e.o.policy.fill_rows = rows;
    var cache: Prefix = .{};
    var h: Host = .{ .e = &e, .lanes = .{ .lane = &lane }, .prefix = &cache };
    const stops = hook and cancel_inside and world == 1 and rows >= 512;
    if (stops) {
        try T.expectError(error.Cancelled, source.Hip.promptStep(&h, &s));
        try T.expectEqual(@as(usize, 0), lane.fill.?.at);
        try T.expectEqual(@as(usize, 0), lane.fill.?.next);
        try T.expectEqual(@as(usize, 0), cache.kept);
        try T.expectEqual(@as(usize, 0), h.taken);
    } else {
        try T.expectEqual(final, try source.Hip.promptStep(&h, &s));
        try T.expectEqual(@as(usize, @intFromBool(final)), h.taken);
        if (final) {
            try T.expect(lane.fill == null);
        } else {
            try T.expectEqual(rows, lane.fill.?.at);
            try T.expectEqual(@as(usize, 1), lane.fill.?.next);
            try T.expectEqual(@as(usize, 1), cache.kept);
        }
    }
    try T.expectEqual(rows, e.rows);
    try T.expectEqual(@as(usize, @intFromBool(!final)), e.advance_calls);
    try T.expectEqual(@as(usize, @intFromBool(final)), e.prefill_calls);
    if (hook) {
        const c = e.forwarded orelse return error.CancelHookNotForwarded;
        try T.expect(c.ctx == @as(*anyopaque, &flag));
        try T.expect(c.check == Flag.check);
    } else try T.expect(e.forwarded == null);
}

test "intermediate prompt forwards the request hook before publishing progress" {
    for ([_]usize{ 511, 512, 1024 }) |rows| {
        for ([_]usize{ 1, 2 }) |world| {
            try run(false, rows, world, true, true);
            try run(false, rows, world, true, false);
            try run(false, rows, world, false, true);
        }
    }
}

test "final prompt forwards the request hook before publishing its first token" {
    for ([_]usize{ 511, 512, 1024 }) |rows| {
        for ([_]usize{ 1, 2 }) |world| {
            try run(true, rows, world, true, true);
            try run(true, rows, world, true, false);
            try run(true, rows, world, false, true);
        }
    }
}
