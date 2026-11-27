//! A hand-off stream whose prompt runs in pieces gets its export point armed at `begin`, run once by the target.
const std = @import("std");
const lanes = @import("lanes");
const twin = @import("twin.zig");
const ln = @import("lanes.zig");
const iface = @import("iface.zig");
const costs_mod = @import("costs.zig");

const gpa = std.testing.allocator;

/// The twin as a target that runs prompts in pieces: its `finish` calls the armed cut before the last token's row.
const Probe = struct {
    inner: *twin.Twin,
    armed: [4]?iface.Target.Cut = @splat(null),
    log: std.ArrayList(u8) = .empty,
    pieces_run: u32 = 0,

    fn target(p: *Probe) iface.Target {
        return .{ .ptr = p, .vtable = &vtable };
    }

    const vtable: iface.Target.VTable = .{ .prefill = prefill, .window = window, .keep = keep, .taps = taps, .release = release, .begin = begin, .piece = piece, .finish = finish, .arm = arm };

    fn prefill(ptr: *anyopaque, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64) anyerror!u32 {
        return self(ptr).inner.target().prefill(slot, ids, smp, draw);
    }

    fn window(ptr: *anyopaque, segments: []const iface.Segment, choices: [][]u32) anyerror!void {
        return self(ptr).inner.target().window(segments, choices);
    }

    fn keep(ptr: *anyopaque, slot: u32, path: []const u32) anyerror!void {
        return self(ptr).inner.target().keep(slot, path);
    }

    fn taps(ptr: *anyopaque, slot: u32) iface.Taps {
        return self(ptr).inner.target().taps(slot);
    }

    fn release(ptr: *anyopaque, slot: u32) void {
        self(ptr).inner.target().release(slot);
    }

    fn self(ptr: *anyopaque) *Probe {
        return @ptrCast(@alignCast(ptr));
    }

    fn arm(ptr: *anyopaque, slot: u32, cut: ?iface.Target.Cut) void {
        self(ptr).armed[slot] = cut;
    }

    fn begin(ptr: *anyopaque, slot: u32, _: []const u32) anyerror!iface.Target.Begun {
        self(ptr).armed[slot] = null; // as GpuTarget.begin: an old prompt's point is not this one's
        return .{};
    }

    fn piece(ptr: *anyopaque, _: u32, _: []const u32, _: u64, _: u64, _: bool) anyerror!void {
        self(ptr).pieces_run += 1;
    }

    fn finish(ptr: *anyopaque, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64) anyerror!u32 {
        const p = self(ptr);
        if (p.armed[slot]) |c| {
            p.armed[slot] = null;
            c.run(c.ptr);
        }
        try p.log.appendSlice(gpa, "finish;");
        return p.inner.target().prefill(slot, ids, smp, draw);
    }
};

const Hook = struct {
    probe: *Probe,
    calls: u32 = 0,
    at: u32 = 0,
    finished_first: bool = false,

    fn at_(ptr: *anyopaque, _: *lanes.Stream, at: u32) void {
        const h: *Hook = @ptrCast(@alignCast(ptr));
        h.calls += 1;
        h.at = at;
        h.finished_first = std.mem.indexOf(u8, h.probe.log.items, "finish;") != null;
    }
};

const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };

fn open(handoff: bool, hook: *Hook, marks: []const u32) !lanes.Stream {
    return lanes.Stream.init(gpa, .{ .id = "s", .prompt = &prompt, .max_new = 1, .drafts = false, .reuse = .{ .marks = if (handoff) marks else &.{}, .hook = if (handoff) .{ .ptr = hook, .at = Hook.at_ } else null } });
}

fn runOne(handoff: bool) !Hook {
    const t = try twin.Twin.init(gpa, .{}, 5);
    defer t.deinit();
    const d = try twin.Drafter.init(gpa, t, 5);
    defer d.deinit();
    var probe: Probe = .{ .inner = t };
    defer probe.log.deinit(gpa);
    const tgt = probe.target();
    var c = try costs_mod.defaults(gpa, 64);
    defer c.deinit(gpa);
    const x = try ln.Lanes.init(gpa, tgt, d.pass(), c, .{ .shape = d.shape, .slots = 4 });
    defer x.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cfg = try lanes.Config.init(gpa, try x.model(arena.allocator(), 64), 16, 15);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpa, &cfg, x.backend(), clock.clock());
    defer e.deinit();
    x.attach(&e);
    var hook: Hook = .{ .probe = &probe };
    const marks = [_]u32{prompt.len - 1};
    var s = try open(handoff, &hook, &marks);
    defer s.deinit(gpa);
    _ = try e.beginStream(&s);
    try e.piece(&s, 0, 8, false);
    try e.piece(&s, 8, prompt.len - 1, false);
    try e.finishStream(&s);
    while (e.activeCount() > 0) try e.step();
    try std.testing.expectEqual(@as(u32, 2), probe.pieces_run);
    return hook;
}

test "a hand-off stream in pieces exports once at its cut, before the last token's row" {
    const h = try runOne(true);
    try std.testing.expectEqual(@as(u32, 1), h.calls);
    try std.testing.expectEqual(@as(u32, prompt.len - 1), h.at);
    try std.testing.expect(!h.finished_first); // the hook ran inside `finish`, before it appended its log line
}

test "a stream that is not a hand-off arms nothing" {
    const h = try runOne(false);
    try std.testing.expectEqual(@as(u32, 0), h.calls);
}
