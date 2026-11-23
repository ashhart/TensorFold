//! The pair's window and head-step ms, timed at load with both Macs running the same windows, kept per build.
const std = @import("std");
const lanes = @import("lanes");
const st = @import("state.zig");
const Slots = @import("slots.zig").Slots;
const Cost = lanes.config.Cost;
const cost_rule = lanes.cost_rule;

pub const Costs = struct { window: [st.max_rows]Cost, head_ms: f64 };

/// What one timing gives: each width's fastest window and the head's fastest 1- and 8-level chains.
pub const Timed = struct { window: [st.max_rows]f64, head: [2]f64 };

/// The scratch stream's prompt; it starts again before passing 112 tokens, inside any context the warm-up fit.
const prompt_len = 64;
const room = 112;
/// Windows of the widest width first: the GPU's clocks ramp up under load.
const warm = 6;
/// Head chains of 1 and 8 levels: their difference over 7 is a level's ms.
const levels = [2]u32{ 1, 8 };

/// The depth rule's costs: timed now (both Macs time, rank 0 keeps them) or kept from an earlier load of this build.
pub fn measure(sl: *Slots, gpa: std.mem.Allocator, io: std.Io) !?Costs {
    const e = sl.e;
    const lead = !e.followsPeer();
    var shape: [96]u8 = undefined;
    const parts = [_][]const u8{ "glm-metal-pair", std.mem.span(e.device.name()), try std.fmt.bufPrint(&shape, "{d}/{d}/{d}/{d}/{d}/{d}", .{ e.c.layers, e.c.hidden, e.c.experts, e.c.tp, e.c.inter[1] - e.c.inter[0], @intFromBool(e.hasMtp()) }) };
    const key = if (lead) lanes.cost_cache.key(gpa, io, &parts) catch null else null;
    var kept = if (key) |k| lanes.cost_cache.load(Costs, gpa, io, k) else null;
    if (kept) |table| for (table.window, 0..) |w, i| {
        if (w.width != i + 1 or !(w.ms > 0) or !std.math.isFinite(w.ms)) {
            kept = null;
            break;
        }
    };
    const cold = try e.agree(kept == null); // the leader decides whether both ranks measure or heal
    var scratch: Scratch = .{ .sl = sl, .io = io };
    defer if (scratch.started) sl.release(0);
    if (!cold) {
        var costs: Costs = if (kept) |c| c else undefined;
        var values: [st.max_rows]f64 = @splat(0);
        var widths: [st.max_rows]u32 = undefined;
        for (&widths, 0..) |*w, i| w.* = @intCast(i + 1);
        if (lead) for (costs.window, &values) |w, *m| {
            m.* = w.ms;
        };
        const cursor_key = if (key) |k| lanes.cost_cache.buildKey(&k, &.{"metal-heal-v2"}) else null;
        var cursor: u64 = if (cursor_key) |k| lanes.cost_cache.load(u64, gpa, io, k) orelse 0 else 0;
        var timer: WarmTimer = .{ .s = &scratch, .wall = .{ .io = io } };
        const stats = try cost_rule.warm_prices.prepared(&timer, &values, &widths, &cursor, 0, Agree{ .slots = sl }, lead);
        if (!lead) return null;
        for (&costs.window, values) |*w, m| w.ms = m;
        if (stats.lowered > 0) if (key) |k| lanes.cost_cache.save(Costs, gpa, io, k, costs);
        if (stats.reads > 0) if (cursor_key) |k| lanes.cost_cache.save(u64, gpa, io, k, cursor);
        report(costs, "checked");
        return costs;
    }
    try scratch.start();
    const ref_key = lanes.cost_cache.referenceKey(&parts);
    const reference = if (lead) lanes.cost_cache.load(Timed, gpa, io, ref_key) else null;
    const t = try schedule(WindowTimer{ .s = &scratch }, HeadTimer{ .s = &scratch }, Agree{ .slots = sl }, lead, e.hasMtp(), reference);
    if (!lead) return null;
    const costs = fromTimed(t);
    if (key) |k| lanes.cost_cache.save(Costs, gpa, io, k, costs);
    lanes.cost_cache.save(Timed, gpa, io, ref_key, t);
    report(costs, "timed");
    return costs;
}

fn report(c: Costs, how: []const u8) void {
    std.log.info("glm: the pair's windows ({s}): {d:.2} {d:.2} {d:.2} {d:.2} {d:.2} ms at 1-5 rows, {d:.2} at 16, a head step {d:.2} ms", .{ how, c.window[0].ms, c.window[1].ms, c.window[2].ms, c.window[3].ms, c.window[4].ms, c.window[15].ms, c.head_ms });
}

/// Both ranks follow the leader's selected repair reads and drift retries, regardless of their own clocks.
pub fn schedule(windows: anytype, heads: anytype, agree: anytype, lead: bool, head: bool, reference: ?Timed) !Timed {
    for (0..warm) |_| _ = try windows.time(st.max_rows - 1);
    var t: Timed = .{ .window = undefined, .head = .{ 0, 0 } };
    try cost_rule.measureWindows(windows, &t.window, false);
    if (head) for (&t.head, 0..) |*h, i| {
        h.* = try cost_rule.fastest(heads, i);
    };
    try cost_rule.smoothSynced(windows, &t.window, agree, lead);
    const drifted = if (reference) |r| cost_rule.drifted(&t.window, &r.window) else false;
    if (try agree.decide(lead and drifted)) {
        try cost_rule.measureWindows(windows, &t.window, true);
        if (head) for (&t.head, 0..) |*h, i| {
            h.* = @min(h.*, try cost_rule.fastest(heads, i));
        };
    }
    return t;
}

/// The depth rule's table: window ms by width, and a head level's ms from the two chains.
pub fn fromTimed(t: Timed) Costs {
    var c: Costs = .{ .window = undefined, .head_ms = @max(0, (t.head[1] - t.head[0]) / @as(f64, @floatFromInt(levels[1] - levels[0]))) };
    for (&c.window, t.window, 0..) |*w, ms, i| w.* = .{ .width = @intCast(i + 1), .ms = ms };
    return c;
}

/// Rank 0's choice, the one both Macs take (one Mac: its own).
const Agree = struct {
    slots: *Slots,
    pub fn decide(a: Agree, mine: bool) !bool {
        return a.slots.e.agree(mine);
    }
};

/// Slot 0's scratch stream: windows over a short prompt, begun again before it passes `room` tokens.
const Scratch = struct {
    sl: *Slots,
    io: std.Io,
    tokens: [prompt_len]u32 = undefined,
    started: bool = false,

    fn start(s: *Scratch) !void {
        for (&s.tokens, 0..) |*t, i| t.* = @intCast(1000 + i * 7919 % 50000);
        try s.sl.begin(0, &s.tokens, true);
        s.started = true;
        var at: u32 = 0;
        while (at < prompt_len) {
            const n = s.sl.chunkRows(at, prompt_len);
            try s.sl.chunk(0, at, n, null);
            at += n;
        }
    }

    /// Room for a window of `st.max_rows` rows: the stream begins again when it has grown too long.
    fn ensureRoom(s: *Scratch) !void {
        if (s.sl.length(0) + st.max_rows + 1 <= room) return;
        s.sl.release(0);
        try s.start();
    }

    /// A window of `rows` rows on slot 0, timed to its picks once the work before it has finished; one row kept.
    fn window(s: *Scratch, rows: usize) !f64 {
        try s.ensureRoom();
        try s.sl.flush();
        var clock: lanes.backend.WallClock = .{ .io = s.io };
        clock.clock().start();
        try s.sl.window(&.{.{ .slot = 0, .pending = s.tokens[0], .held = 0, .tokens = s.tokens[1..rows] }}, null);
        const ms = clock.clock().elapsedMs(.cost);
        try s.sl.keep(0, 1);
        return ms;
    }
};

const WindowTimer = struct {
    s: *Scratch,
    pub fn time(t: WindowTimer, i: usize) !f64 {
        const ms = try t.s.window(i + 1);
        if (t.s.sl.slots[0].mtp) try t.s.sl.draftAll(&.{.{ .slot = 0, .prompt = false, .follow = t.s.tokens[1..2], .depth = 0 }});
        return ms;
    }
};

const WarmTimer = struct {
    s: *Scratch,
    wall: lanes.backend.WallClock,
    pub fn start(t: *WarmTimer) void {
        t.wall.clock().start();
    }
    pub fn elapsedMs(t: *WarmTimer) f64 {
        return t.wall.clock().elapsedMs(.cost);
    }
    pub fn prepare(t: *WarmTimer, _: anytype, _: []const u32) !void {
        try t.s.start();
    }
    pub fn time(t: *WarmTimer, i: usize) !f64 {
        const ms = try (WindowTimer{ .s = t.s }).time(i);
        try t.s.sl.flush();
        return ms;
    }
};

const HeadTimer = struct {
    s: *Scratch,
    pub fn time(t: HeadTimer, i: usize) !f64 {
        _ = try t.s.window(1);
        try t.s.sl.flush();
        var clock: lanes.backend.WallClock = .{ .io = t.s.io };
        clock.clock().start();
        try t.s.sl.draftAll(&.{.{ .slot = 0, .prompt = false, .follow = t.s.tokens[1..2], .depth = levels[i] }});
        try t.s.sl.flush();
        return clock.clock().elapsedMs(.cost);
    }
};

/// Scripted clocks for the tests: width i reads 10 + i ms, or 10 + i - 3 at `dip`.
const Script = struct {
    calls: *std.ArrayList(usize),
    dip: ?usize = null,
    pub fn time(s: Script, i: usize) !f64 {
        try s.calls.append(std.testing.allocator, i);
        return 10 + @as(f64, @floatFromInt(i)) - @as(f64, if (s.dip == i) 3 else 0);
    }
};

const Fixed = struct {
    answer: bool,
    asked: *?bool,
    pub fn decide(f: Fixed, mine: bool) !bool {
        f.asked.* = mine;
        return f.answer;
    }
};

test "both Macs make the same timing calls, and only rank 0's word starts a second pass" {
    const gpa = std.testing.allocator;
    var runs: [2]std.ArrayList(usize) = .{ .empty, .empty };
    defer for (&runs) |*r| r.deinit(gpa);
    // rank 0 reads a dip at 3 rows; rank 1's clock reads nothing odd but follows rank 0's word
    for ([_]bool{ true, false }, &runs) |lead, *calls| {
        var asked: ?bool = null;
        const w: Script = .{ .calls = calls, .dip = if (lead) 3 else null };
        const t = try schedule(w, w, Fixed{ .answer = true, .asked = &asked }, lead, true, null);
        try std.testing.expectEqual(false, asked.?);
        try std.testing.expect(t.window[0] == 10 and t.head[1] > t.head[0]);
    }
    try std.testing.expectEqualSlices(usize, runs[0].items, runs[1].items);
    // The forced decisions add one repair per width, then a complete retry with the same head samples.
    try std.testing.expectEqual(@as(usize, warm + st.max_rows + 2 * (st.max_rows + 2) * (1 + cost_rule.reps)), runs[0].items.len);
}

test "the table holds every width and a head level's ms from the two chains" {
    var t: Timed = .{ .window = undefined, .head = .{ 1.5, 5.0 } };
    for (&t.window, 0..) |*m, i| m.* = 10 + 3 * @as(f64, @floatFromInt(i));
    const c = fromTimed(t);
    try std.testing.expectEqual(@as(u32, 1), c.window[0].width);
    try std.testing.expectEqual(@as(u32, 16), c.window[15].width);
    try std.testing.expectEqual(@as(f64, 55), c.window[15].ms);
    try std.testing.expectEqual(@as(f64, 0.5), c.head_ms);
    try std.testing.expect(cost_rule.repairAt(&.{ 10, 13, 12.5, 16 }, 2));
    try std.testing.expect(!cost_rule.repairAt(&.{ 10, 13, 12.7, 16 }, 2));
}
