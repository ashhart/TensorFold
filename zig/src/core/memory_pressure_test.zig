//! Memory pressure: the governor's hysteresis and settings, and the prompt cache's shedding and hold.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const mp = @import("memory_pressure.zig");
const pc = @import("prompt_cache.zig");
const shed = @import("prompt_cache_shed.zig").shed;
const Fake = @import("prompt_cache_fake.zig").Fake;
const Level = mp.Level;
const Store = pc.Store;
const s_ns: i96 = std.time.ns_per_s;

/// A pressure source a test sets, counting its reads.
const Dial = struct {
    level: ?Level = .normal,
    reads: usize = 0,
    fn read(ctx: ?*anyopaque) ?Level {
        const d: *Dial = @ptrCast(@alignCast(ctx.?));
        d.reads += 1;
        return d.level;
    }
    fn source(d: *Dial) mp.Source {
        return .{ .ctx = d, .read = read };
    }
};

test "memory pressure: a rise takes effect at once, a fall only after a calm spell, each change reported once" {
    var dial: Dial = .{};
    var g: mp.Governor = .{ .source = dial.source(), .every_ns = 0 };
    const Step = struct { at_ms: i96, raw: Level, governed: Level, change: bool };
    const steps = [_]Step{
        .{ .at_ms = 0, .raw = .normal, .governed = .normal, .change = false },
        .{ .at_ms = 1000, .raw = .warning, .governed = .warning, .change = true },
        .{ .at_ms = 1250, .raw = .warning, .governed = .warning, .change = false },
        .{ .at_ms = 2000, .raw = .normal, .governed = .warning, .change = false }, // a calm spell starts
        .{ .at_ms = 3000, .raw = .warning, .governed = .warning, .change = false }, // and ends: a flap moves nothing
        .{ .at_ms = 4000, .raw = .normal, .governed = .warning, .change = false },
        .{ .at_ms = 8999, .raw = .normal, .governed = .warning, .change = false },
        .{ .at_ms = 9000, .raw = .normal, .governed = .normal, .change = true }, // five calm seconds
        .{ .at_ms = 10000, .raw = .critical, .governed = .critical, .change = true },
        .{ .at_ms = 11000, .raw = .warning, .governed = .critical, .change = false },
        .{ .at_ms = 16000, .raw = .warning, .governed = .warning, .change = true },
        .{ .at_ms = 17000, .raw = .critical, .governed = .critical, .change = true },
        .{ .at_ms = 18000, .raw = .normal, .governed = .critical, .change = false },
        .{ .at_ms = 20000, .raw = .warning, .governed = .critical, .change = false }, // still below: the spell goes on
        .{ .at_ms = 23000, .raw = .normal, .governed = .normal, .change = true },
        .{ .at_ms = 24000, .raw = .normal, .governed = .normal, .change = false },
    };
    var changes: usize = 0;
    for (steps) |st| {
        dial.level = st.raw;
        const c = g.poll(st.at_ms * std.time.ns_per_ms);
        try std.testing.expectEqual(st.governed, g.level);
        try std.testing.expectEqual(st.change, c != null);
        if (c) |x| {
            try std.testing.expect(x.from != x.to);
            try std.testing.expectEqual(st.governed, x.to);
            changes += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 6), changes);
}

test "memory pressure: the kernel is read at most once an interval, and no reading counts as normal" {
    var dial: Dial = .{ .level = .warning };
    var g: mp.Governor = .{ .source = dial.source(), .calm_ns = 0 };
    for ([_]i96{ 0, 100, 249, 250, 400, 500 }) |ms| _ = g.poll(ms * std.time.ns_per_ms);
    try std.testing.expectEqual(@as(usize, 3), dial.reads);
    try std.testing.expectEqual(Level.warning, g.level);
    dial.level = null; // the sysctl failed
    _ = g.poll(s_ns);
    try std.testing.expectEqual(Level.normal, g.level);
}

test "memory pressure: settings, the watermark and the kernel's levels" {
    try std.testing.expectEqual(mp.Settings{ .on = true, .percent = 50 }, mp.Settings.fromText(null, null));
    for ([_][]const u8{ "0", "off", "OFF", " off " }) |t| try std.testing.expect(!mp.Settings.fromText(t, null).on);
    for ([_][]const u8{ "1", "on", "" }) |t| try std.testing.expect(mp.Settings.fromText(t, null).on);
    for ([_]struct { []const u8, u8 }{ .{ "30", 30 }, .{ "0", 0 }, .{ "100", 100 }, .{ "101", 50 }, .{ "-5", 50 }, .{ "x", 50 } }) |c|
        try std.testing.expectEqual(c[1], mp.Settings.fromText(null, c[0]).percent);
    var dial: Dial = .{};
    var g: mp.Governor = .{ .source = dial.source(), .percent = 50 };
    try std.testing.expectEqual(@as(u64, 500), g.mark(1000));
    g.percent = 0;
    try std.testing.expectEqual(@as(u64, 0), g.mark(1000));
    g.percent = 100;
    try std.testing.expectEqual(@as(u64, std.math.maxInt(u64)), g.mark(std.math.maxInt(u64)));
    try std.testing.expectEqual(Level.normal, mp.ofKernel(1));
    try std.testing.expectEqual(Level.warning, mp.ofKernel(2));
    try std.testing.expectEqual(Level.critical, mp.ofKernel(4));
    try std.testing.expect(mp.Governor.fromEnv() == null); // host tests never read the kernel's level
    const raw = platform.memory_policy.pressure();
    if (builtin.os.tag == .macos) try std.testing.expect(raw != null) else try std.testing.expect(raw == null);
}

const c1 = [_]u32{ 1, 1, 1, 1, 9 };
const c1_next = [_]u32{ 1, 1, 1, 1, 9, 5, 5, 9 };
const c2 = [_]u32{ 2, 2, 2, 2, 2, 2, 9 };
const c3 = [_]u32{ 3, 3, 3, 3, 3, 3, 3, 3, 9 };
const c3_next = [_]u32{ 3, 3, 3, 3, 3, 3, 3, 3, 9, 4, 9 };

fn ats(s: *const Store, out: []u32) []u32 {
    for (s.entries.items, 0..) |e, i| out[i] = e.at;
    std.mem.sort(u32, out[0..s.entries.items.len], {}, std.sort.asc(u32));
    return out[0..s.entries.items.len];
}

fn fresh(prompt: []const u32) u64 {
    var sum: u64 = 0;
    for (prompt) |t| sum = sum *% 31 +% t;
    return sum;
}

test "memory pressure: the prompt cache sheds to its watermark least valuable first, never a borrowed state" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    _ = f.pass(&s, &c1, try s.begin(a, &c1, 4, &.{}, &.{}, null, &.{}));
    _ = f.pass(&s, &c2, try s.begin(a, &c2, 6, &.{}, &.{}, null, &.{}));
    _ = f.pass(&s, &c3, try s.begin(a, &c3, 8, &.{}, &.{}, null, &.{}));
    _ = f.pass(&s, &c1_next, try s.begin(a, &c1_next, 7, &.{}, &.{}, null, &.{}));
    var buf: [8]u32 = undefined;
    try std.testing.expectEqualSlices(u32, &.{ 4, 6, 7, 8 }, ats(&s, &buf));
    try std.testing.expect(!shed(&s, s.held)); // at the watermark: nothing goes
    const l = try s.lookup(a, &c3_next, 9, &.{}, &.{}, &.{}); // c3's next turn borrows its state at 8
    while (shed(&s, 220)) {}
    try std.testing.expectEqualSlices(u32, &.{ 7, 8 }, ats(&s, &buf)); // c1 at 4 (moved past), then c2 (oldest)
    try std.testing.expectEqual(@as(u64, 107 + 108), s.held);
    while (shed(&s, 0)) {}
    try std.testing.expectEqualSlices(u32, &.{8}, ats(&s, &buf)); // the borrowed state stays past the watermark
    try std.testing.expectEqual(@as(u64, 3), s.counts.evicted);
    try std.testing.expectEqual(@as(usize, 1), f.live);
    s.resumed(l.entry.?, &c3_next, true);
    try std.testing.expect(shed(&s, 0)); // returned: it may go now
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
    try std.testing.expectEqual(@as(usize, 0), f.live);
}

test "memory pressure: spare storage past the watermark is trimmed after the states" {
    const gpa = std.testing.allocator;
    var f: Fake = .{ .gpa = gpa, .spare_bytes = 1000 };
    var s = Store.init(gpa, f.pooled(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    f.at = 4;
    try std.testing.expect(s.keep(&c1, 4, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(u64, 94), s.held);
    try std.testing.expectEqual(@as(u64, 906), s.spare());
    try std.testing.expect(!shed(&s, 500)); // the state fits: only the spare past what it leaves goes
    try std.testing.expectEqual(@as(u64, 406), s.spare());
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expect(shed(&s, 0)); // the state goes, and the storage it gave back as spare with it
    try std.testing.expectEqual(@as(u64, 0), s.held);
    try std.testing.expectEqual(@as(u64, 0), s.spare());
}

test "memory pressure: a held store keeps nothing new and plans no marks, yet still resumes exactly" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    _ = f.pass(&s, &c1, try s.begin(a, &c1, 4, &.{}, &.{}, null, &.{}));
    s.hold = true;
    const p = try s.begin(a, &c1_next, 7, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 4), p.from); // a kept state is still resumed
    try std.testing.expectEqual(@as(usize, 0), p.marks.len);
    try std.testing.expectEqual(fresh(&c1_next), f.pass(&s, &c1_next, p));
    f.at = 7;
    try std.testing.expect(!s.keep(&c1_next, 7, null, &.{}, &.{})); // a keep the pass asks for anyway is refused
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expectEqual(@as(u64, 1), s.counts.kept);
    s.hold = false;
    const q = try s.begin(a, &c2, 6, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{6}, q.marks);
    try std.testing.expectEqual(fresh(&c2), f.pass(&s, &c2, q));
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
}
