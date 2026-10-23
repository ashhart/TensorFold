//! CPU records through the production prompt route: the shared serving vtable the lane host calls, no HIP driver loaded.
const std = @import("std");
const hip = @import("hip");
const abi = @import("hip").abi;
const lanes = @import("lanes");
const serve = @import("hip_serve");
const source = @import("hip_lanes.zig");
const prefix = @import("prefix.zig");
const T = std.testing;

/// The engine's own cancel shape, which the seam forwards without ever naming the type.
const Cancel = struct {
    ctx: *anyopaque,
    check: *const fn (ctx: *anyopaque) bool,
};

const Flag = struct {
    on: bool = false,

    fn check(ptr: *anyopaque) bool {
        const flag: *Flag = @ptrCast(@alignCast(ptr));
        return flag.on;
    }
};

/// The engine the route forwards to, recorded instead of run: it applies the same poll policy `Engine.poll` does.
const Record = struct {
    world: usize = 1,
    cancel_inside: bool = false,
    flag: *Flag,
    advance_calls: usize = 0,
    prefill_calls: usize = 0,
    forwarded: ?Cancel = null,
    rows: usize = 0,

    fn capture(e: *Record, rows: usize, cancel: ?Cancel) !void {
        e.forwarded = cancel;
        e.rows = rows;
        if (e.cancel_inside) e.flag.on = true;
        // The engine polls after each layer only for a single rank and a long enough chunk.
        if (prefix.pollsLayers(e.world, rows)) {
            if (cancel) |c| if (c.check(c.ctx)) return error.Cancelled;
        }
    }
};

/// The lane the seam holds per stream: the pass in progress, the kept row a grammar draws from, the held drafts.
const Lane = struct {
    id: u32 = 0,
    fill: ?serve.backend.Fill = null,
    first_row: ?[]u16 = null,
    pending: ?struct { rows: usize } = null,
    held: [16]u32 = @splat(0),
    held_n: usize = 0,
    len: usize = 0,
};

const Drawer = struct {
    vocab: usize = 8,

    pub fn drawHost(_: *Drawer, _: []const u16, _: anytype) !u32 {
        return 0;
    }
};

fn setCurrent(_: abi.Context) callconv(.c) abi.Result {
    return 0;
}

/// A runtime whose only answer is the current-context call the seam's `of` makes before a chunk.
const inert = struct {
    var runtime: hip.runtime.Runtime = .{ .lib = undefined, .api = undefined };
};

/// The backend the seam drives, recording what it is handed. Same surface as `Hip`, so the route under test is
/// the production one and only the engine underneath is inert.
const Fake = struct {
    gpa: std.mem.Allocator,
    e: struct { ctx: hip.Context },
    streams: serve.backend.Streams(Lane),
    tp: serve.protocol.Sender,
    rec: *Record,
    lane: *Lane = undefined,
    drawer_state: Drawer = .{},
    fill_rows: usize = 512,
    kept: usize = 0,
    rows_kept: usize = 0,
    drafting: usize = 0,
    verifying: usize = 0,
    flushed: usize = 0,
    released: usize = 0,

    pub fn drawer(h: *Fake) *Drawer {
        return &h.drawer_state;
    }

    pub fn cancelOf(_: *Fake, s: *lanes.Stream) ?Cancel {
        const c = s.cancel_check orelse return null;
        return .{ .ctx = c.ptr, .check = c.check };
    }

    pub fn begin(h: *Fake, s: *lanes.Stream) !*Lane {
        h.lane.fill = serve.backend.Fill.init(0, &.{@intCast(h.fill_rows)});
        h.lane.len = s.prompt_len;
        const id = try h.streams.idOf(s);
        h.lane.id = id;
        try h.streams.map.put(h.gpa, s, h.lane);
        return h.lane;
    }

    pub fn reach(h: *Fake, _: *Lane, at: usize, limit: usize) !usize {
        return prefix.chunkEnd(at, limit, h.fill_rows);
    }

    pub fn advance(h: *Fake, lane: *Lane, prompt: []const u32, from: usize, to: usize, cancel: ?Cancel) !void {
        _ = lane;
        _ = prompt;
        h.rec.advance_calls += 1;
        try h.rec.capture(to - from, cancel);
    }

    pub fn finish(h: *Fake, _: *Lane, prompt: []const u32, from: usize, _: anytype, cancel: ?Cancel) !u32 {
        h.rec.prefill_calls += 1;
        try h.rec.capture(prompt.len - from, cancel);
        return 7;
    }

    pub fn keepCut(h: *Fake, _: *Lane, _: []const u32, _: usize) !void {
        h.kept += 1;
    }

    pub fn keepRows(h: *Fake, _: *Lane, rows: usize) !void {
        h.rows_kept += rows;
    }

    pub fn flush(h: *Fake) !void {
        h.flushed += 1;
    }

    pub fn verify(h: *Fake, _: []const lanes.backend.Window, _: []lanes.backend.Verified) !void {
        h.verifying += 1;
    }

    pub fn draft(h: *Fake, _: []const lanes.backend.DraftRequest) !void {
        h.drafting += 1;
    }

    pub fn release(h: *Fake, _: *Lane) void {
        h.released += 1;
    }

    fn decoded() bool {
        return true;
    }
};

const Shared = serve.backend.Backend(Fake);

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

    var lane: Lane = .{};
    var rec: Record = .{ .flag = &flag, .world = world, .cancel_inside = cancel_inside };
    inert.runtime.api.hipCtxSetCurrent = setCurrent;
    var fake: Fake = .{
        .gpa = T.allocator,
        .e = .{ .ctx = .{ .r = &inert.runtime, .device = 0, .handle = null } },
        .streams = serve.backend.Streams(Lane).init(T.allocator),
        .tp = .{ .gpa = T.allocator },
        .rec = &rec,
        .lane = &lane,
        .fill_rows = rows,
    };
    defer fake.streams.deinit();

    const route = Shared.backend(&fake);
    const step = route.vtable.prefill_step.?;

    // The engine polls only for one rank and a chunk of at least 512 rows, so only that combination stops.
    const stops = hook and cancel_inside and world == 1 and rows >= 512;
    if (stops) {
        try T.expectError(error.Cancelled, step(fake_ptr(&fake), &s));
        try T.expectEqual(@as(usize, 0), lane.fill.?.at);
        try T.expectEqual(@as(usize, 0), lane.fill.?.next);
        try T.expectEqual(@as(usize, 0), fake.kept);
    } else {
        try T.expectEqual(final, try step(fake_ptr(&fake), &s));
        if (final) {
            try T.expect(lane.fill == null);
        } else {
            try T.expectEqual(rows, lane.fill.?.at);
            try T.expectEqual(@as(usize, 1), lane.fill.?.next);
            try T.expectEqual(@as(usize, 1), fake.kept);
        }
    }
    try T.expectEqual(rows, rec.rows);
    try T.expectEqual(@as(usize, @intFromBool(!final)), rec.advance_calls);
    try T.expectEqual(@as(usize, @intFromBool(final)), rec.prefill_calls);
    // The route hands the engine the stream's own hook on both the intermediate and the final chunk.
    if (hook) {
        const c = rec.forwarded orelse return error.CancelHookNotForwarded;
        try T.expect(c.ctx == @as(*anyopaque, &flag));
        try T.expect(c.check == Flag.check);
    } else try T.expect(rec.forwarded == null);
}

fn fake_ptr(f: *Fake) *anyopaque {
    return @ptrCast(f);
}

test "the shared route forwards the request hook to the engine on an intermediate chunk" {
    for ([_]usize{ 511, 512, 1024 }) |rows| {
        for ([_]usize{ 1, 2 }) |world| {
            try run(false, rows, world, true, true);
            try run(false, rows, world, true, false);
            try run(false, rows, world, false, true);
        }
    }
}

test "the shared route forwards the request hook on the final chunk, before the first token" {
    for ([_]usize{ 511, 512, 1024 }) |rows| {
        for ([_]usize{ 1, 2 }) |world| {
            try run(true, rows, world, true, true);
            try run(true, rows, world, true, false);
            try run(true, rows, world, false, true);
        }
    }
}

test "the production hook is the stream's own cancel check" {
    var flag: Flag = .{};
    var four: [4]u32 = @splat(1);
    var with = try lanes.Stream.init(T.allocator, .{ .id = "p", .prompt = &four, .max_new = 4, .cancel_check = .{ .ptr = &flag, .check = Flag.check } });
    defer with.deinit(T.allocator);
    var without = try lanes.Stream.init(T.allocator, .{ .id = "q", .prompt = &four, .max_new = 4 });
    defer without.deinit(T.allocator);
    var engine: source.Hip = undefined;
    // cancelOf reads the stream only, so the receiver is never touched.
    const c = source.Hip.cancelOf(&engine, &with);
    try T.expect(c != null);
    try T.expectEqual(@as(*anyopaque, &flag), c.?.ctx);
    try T.expect(c.?.check == Flag.check);
    try T.expect(source.Hip.cancelOf(&engine, &without) == null);
}
