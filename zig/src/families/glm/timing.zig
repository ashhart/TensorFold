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
/// Depths priced at timed costs sit within a few percent: the acceptance estimates move slower so noise picks no depth.
pub const depth_rate = 0.08;

/// The depth rule's costs: timed now (both Macs time, rank 0 keeps them) or kept from an earlier load of this build.
pub fn measure(sl: *Slots, gpa: std.mem.Allocator, io: std.Io) !?Costs {
    const e = sl.e;
    const lead = !e.followsPeer();
    var shape: [96]u8 = undefined;
    const parts = [_][]const u8{ "glm-metal-pair", std.mem.span(e.device.name()), try std.fmt.bufPrint(&shape, "{d}/{d}/{d}/{d}/{d}/{d}", .{ e.c.layers, e.c.hidden, e.c.experts, e.c.tp, e.c.inter[1] - e.c.inter[0], @intFromBool(e.hasMtp()) }) };
    const key = if (lead) lanes.cost_cache.key(gpa, io, &parts) catch null else null;
    const kept = if (key) |k| lanes.cost_cache.load(Costs, gpa, io, k) else null;
    if (!try e.agree(kept == null)) { // rank 0's word: both Macs time, or neither does
        if (kept) |c| report(c, "kept");
        return kept;
    }
    var scratch: Scratch = .{ .sl = sl, .io = io };
    try scratch.start();
    defer sl.release(0);
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

/// The same calls on both Macs whatever their clocks read: rank 0's dip or drift alone decides a second pass.
pub fn schedule(windows: anytype, heads: anytype, agree: anytype, lead: bool, head: bool, reference: ?Timed) !Timed {
    for (0..warm) |_| _ = try windows.time(st.max_rows - 1);
    var t: Timed = .{ .window = undefined, .head = .{ 0, 0 } };
    for (&t.window, 0..) |*m, i| m.* = try cost_rule.fastest(windows, i);
    if (head) for (&t.head, 0..) |*h, i| {
        h.* = try cost_rule.fastest(heads, i);
    };
    const drifted = if (reference) |r| cost_rule.drifted(&t.window, &r.window) else false;
    if (try agree.decide(lead and (dipped(&t.window) or drifted))) {
        try cost_rule.again(windows, &t.window);
        if (head) for (&t.head, 0..) |*h, i| {
            h.* = @min(h.*, try cost_rule.fastest(heads, i));
        };
    }
    return t;
}

fn dipped(ms: []const f64) bool {
    for (ms[1..], ms[0 .. ms.len - 1]) |m, before| if (m < before * (1 - cost_rule.dip)) return true;
    return false;
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

    fn start(s: *Scratch) !void {
        for (&s.tokens, 0..) |*t, i| t.* = @intCast(1000 + i * 7919 % 50000);
        try s.sl.begin(0, &s.tokens, true);
        var at: u32 = 0;
        while (at < prompt_len) {
            const n = s.sl.chunkRows(at, prompt_len);
            try s.sl.chunk(0, at, n);
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
        try s.sl.window(&.{.{ .slot = 0, .pending = s.tokens[0], .held = 0, .tokens = s.tokens[1..rows] }});
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
        try std.testing.expectEqual(lead, asked.?);
        try std.testing.expect(t.window[0] == 10 and t.head[1] > t.head[0]);
    }
    try std.testing.expectEqualSlices(usize, runs[0].items, runs[1].items);
    // warm-up, 16 widths and 2 head chains of 1 + 7 runs each, then the same again for the second pass
    try std.testing.expectEqual(@as(usize, warm + 2 * (st.max_rows + 2) * (1 + cost_rule.reps)), runs[0].items.len);
}

test "the table holds every width and a head level's ms from the two chains" {
    var t: Timed = .{ .window = undefined, .head = .{ 1.5, 5.0 } };
    for (&t.window, 0..) |*m, i| m.* = 10 + 3 * @as(f64, @floatFromInt(i));
    const c = fromTimed(t);
    try std.testing.expectEqual(@as(u32, 1), c.window[0].width);
    try std.testing.expectEqual(@as(u32, 16), c.window[15].width);
    try std.testing.expectEqual(@as(f64, 55), c.window[15].ms);
    try std.testing.expectEqual(@as(f64, 0.5), c.head_ms);
    try std.testing.expect(dipped(&.{ 10, 13, 12.5, 16 }));
    try std.testing.expect(!dipped(&.{ 10, 13, 12.7, 16 }));
}
