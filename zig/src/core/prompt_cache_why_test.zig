//! #3686: each reason a lookup gives, on the fake running-sum family; every pass still equals a fresh one.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const why = @import("prompt_cache_why.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const fresh = @import("prompt_cache_test.zig").fresh;
const Image = @import("lanes").stream.Image;
const t = std.testing;
const gpa = t.allocator;

const words = blk: {
    var w: [100]u32 = undefined;
    for (&w, 1..) |*x, i| x.* = i;
    break :blk w;
};
const others = blk: {
    var w: [100]u32 = undefined;
    for (&w, 1..) |*x, i| x.* = 1000 + i;
    break :blk w;
};

/// `words` with token `at` changed.
fn edited(at: usize) [100]u32 {
    var w = words;
    w[at] = 999;
    return w;
}

const Box = struct {
    f: Fake = .{ .gpa = gpa },
    s: pc.Store = undefined,
    arena: std.heap.ArenaAllocator = .init(gpa),

    fn init(box: *Box, rules: pc.Rules, budget: u64) void {
        box.s = pc.Store.init(gpa, box.f.snapshots(), rules, budget);
    }

    fn deinit(box: *Box) void {
        box.s.deinit();
        box.arena.deinit();
    }

    /// A pass standing at `at` of `prompt` keeps its state there (`spans`: its decoded rows).
    fn keep(box: *Box, prompt: []const u32, at: u32, spans: []const [2]u32) bool {
        _ = box.f.pass(&box.s, prompt[0..at], .{});
        return box.s.keep(prompt, at, null, &.{}, spans);
    }

    /// `prompt` begun on `starts` and passed: the pass equals a fresh one, and the plan says why.
    fn begin(box: *Box, prompt: []const u32, starts: []const u32) !why.Why {
        const plan = try box.s.begin(box.arena.allocator(), prompt, 0, &.{}, starts, null, &.{});
        try t.expectEqual(fresh(prompt), box.f.pass(&box.s, prompt, plan));
        try t.expectEqual(plan.from, plan.why.resumed);
        return plan.why;
    }
};

fn expectWhy(reason: why.Reason, shared: u32, resumed: u32, got: why.Why) !void {
    try t.expectEqual(reason, got.reason);
    try t.expectEqual(shared, got.shared);
    try t.expectEqual(resumed, got.resumed);
}

const open: pc.Rules = .{ .min_prompt = 0, .min_gap = 0 };

test "prompt_cache why: cold, too_short and bypass" {
    var box: Box = .{};
    box.init(open, 1 << 20);
    defer box.deinit();
    try expectWhy(.cold, 0, 0, try box.begin(&words, &.{}));
    var short: Box = .{};
    short.init(.{}, 1 << 20); // the default shortest kept prompt, 4096 tokens
    defer short.deinit();
    try expectWhy(.too_short, 0, 0, try short.begin(words[0..50], &.{}));
    try t.expectEqual(why.Reason.bypass, (pc.Plan{}).why.reason); // no lookup ran
    try t.expectEqual(why.Reason.bypass, (pc.Lookup{}).why.reason);
}

test "prompt_cache why: none when resumed within SLACK of everything shared, between_marks past it" {
    var box: Box = .{};
    box.init(open, 1 << 20);
    defer box.deinit();
    try t.expect(box.keep(&words, 60, &.{}));
    try expectWhy(.none, 80, 60, try box.begin(words[0..80], &.{})); // 20 more tokens shared: the generation prompt
    var far: Box = .{};
    far.init(open, 1 << 20);
    defer far.deinit();
    try t.expect(far.keep(&words, 20, &.{}));
    const w = try far.begin(&words, &.{});
    try expectWhy(.between_marks, 100, 20, w);
    try t.expectEqual(@as(u32, 80), w.lost());
}

test "prompt_cache why: prompt_changed at the edit, reply_changed inside a reply the engine wrote" {
    var box: Box = .{};
    box.init(open, 1 << 20);
    defer box.deinit();
    try t.expect(box.keep(&words, 20, &.{}));
    try expectWhy(.prompt_changed, 60, 20, try box.begin(&edited(60), &.{}));
    var reply: Box = .{};
    reply.init(open, 1 << 20);
    defer reply.deinit();
    try t.expect(reply.keep(&words, 20, &.{}));
    _ = reply.f.pass(&reply.s, words[0..90], .{});
    try t.expect(reply.s.keepBorn(&words, 50, 90, null, &.{}, &.{})); // a 50-token prompt and its reply
    try expectWhy(.reply_changed, 70, 20, try reply.begin(&edited(70), &.{}));
}

test "prompt_cache why: ends_inside, off_grid and rows_changed name a matching state a reuse check refused" {
    var box: Box = .{};
    box.init(open, 1 << 20);
    defer box.deinit();
    try t.expect(box.keep(&words, 90, &.{}));
    try expectWhy(.ends_inside, 60, 0, try box.begin(words[0..60], &.{}));
    var grid: Box = .{};
    grid.init(.{ .min_prompt = 0, .min_gap = 0, .planned = true }, 1 << 20);
    defer grid.deinit();
    try t.expect(grid.keep(&words, 48, &.{}));
    try expectWhy(.off_grid, 48, 0, try grid.begin(&words, &.{ 32, 64 }));
    var rows: Box = .{};
    rows.init(open, 1 << 20);
    defer rows.deinit();
    try t.expect(rows.keep(&words, 60, &.{.{ 10, 20 }})); // kept with rows 10..20 decoded
    try expectWhy(.rows_changed, 60, 0, try rows.begin(&words, &.{}));
}

test "prompt_cache why: frame_changed and image_changed, by identity and by a re-sized image's tokens" {
    var box: Box = .{};
    box.init(open, 1 << 20);
    defer box.deinit();
    const a = box.arena.allocator();
    _ = box.f.pass(&box.s, words[0..60], .{});
    try t.expect(box.s.keepBornFrame(&words, 100, 60, null, &.{}, &.{}, &.{}, 42));
    const frame = try box.s.lookupUserFrame(a, &words, 100, 0, 0, 0, &.{}, &.{}, &.{}, &.{}, 43);
    try expectWhy(.frame_changed, 60, 0, frame.why);
    const picture: Image = .{ .pixels = &.{}, .gh = 2, .gw = 2, .at = 10, .tokens = 4, .hash = 7 };
    var other = picture;
    other.hash = 8;
    var images: Box = .{};
    images.init(open, 1 << 20);
    defer images.deinit();
    _ = images.f.pass(&images.s, words[0..60], .{});
    try t.expect(images.s.keepBornImages(&words, 100, 60, null, &.{}, &.{}, &.{picture}));
    const swapped = try images.s.lookupUserImages(a, &words, 100, 0, 0, 0, &.{}, &.{}, &.{}, &.{other});
    try expectWhy(.image_changed, 60, 0, swapped.why);
    var resized: Box = .{};
    resized.init(open, 1 << 20);
    defer resized.deinit();
    _ = resized.f.pass(&resized.s, words[0..20], .{});
    try t.expect(resized.s.keepBornImages(&words, 100, 20, null, &.{}, &.{}, &.{}));
    const bigger: Image = .{ .pixels = &.{}, .gh = 2, .gw = 3, .at = 58, .tokens = 6, .hash = 9 };
    const l = try resized.s.lookupUserImages(a, &edited(60), 100, 0, 0, 0, &.{}, &.{}, &.{}, &.{bigger});
    defer if (l.entry) |e| resized.s.release(e);
    try expectWhy(.image_changed, 60, 20, l.why); // the prompt parts inside the image's rows
}

test "prompt_cache why: evicted, shed, expired and cleared states are named, not cold" {
    var box: Box = .{};
    box.init(open, 300); // room for one 160-byte state
    defer box.deinit();
    try t.expect(box.keep(&words, 60, &.{}));
    try t.expect(box.keep(&others, 60, &.{})); // evicts it for room
    try expectWhy(.evicted, 60, 0, try box.begin(&words, &.{}));
    var shed: Box = .{};
    shed.init(open, 1 << 20);
    defer shed.deinit();
    try t.expect(shed.keep(&words, 60, &.{}));
    try t.expect(@import("prompt_cache_shed.zig").shed(&shed.s, 0));
    try expectWhy(.shed, 60, 0, try shed.begin(&words, &.{}));
    var gone: Box = .{};
    gone.init(open, 1 << 20);
    defer gone.deinit();
    try t.expect(gone.keep(&words, 60, &.{}));
    gone.s.expired(gone.s.entries.items[0].saved); // the backend dropped it
    try expectWhy(.expired, 60, 0, try gone.begin(&words, &.{}));
    var cleared: Box = .{};
    cleared.init(open, 1 << 20);
    defer cleared.deinit();
    try t.expect(cleared.keep(&words, 60, &.{}));
    cleared.s.clear();
    try expectWhy(.cleared, 60, 0, try cleared.begin(&words, &.{}));
}

test "prompt_cache why: not_kept under memory pressure and past the budget" {
    var box: Box = .{};
    box.init(open, 1 << 20);
    defer box.deinit();
    box.s.hold = true;
    try t.expect(!box.keep(&words, 60, &.{}));
    box.s.hold = false;
    try expectWhy(.not_kept, 60, 0, try box.begin(&words, &.{}));
    var small: Box = .{};
    small.init(open, 100); // 160 bytes pass the budget
    defer small.deinit();
    try t.expect(!small.keep(&words, 60, &.{}));
    try expectWhy(.not_kept, 60, 0, try small.begin(&words, &.{}));
}

test "prompt_cache why: restore_failed, from the store's retry and from fewer tokens resumed than planned" {
    var box: Box = .{};
    box.init(open, 1 << 20);
    defer box.deinit();
    try t.expect(box.keep(&words, 60, &.{}));
    box.f.refuse_at = 60;
    try expectWhy(.restore_failed, 60, 0, try box.begin(&words, &.{}));
    const planned: why.Why = .{ .reason = .none, .shared = 62, .resumed = 60 };
    try expectWhy(.restore_failed, 62, 0, planned.ended(0)); // a lane backend that restored nothing
    try expectWhy(.none, 62, 60, planned.ended(60));
}

test "prompt_cache why: the ring keeps the newest 64 states in under 3 KiB, and the same state again is renamed" {
    var box: Box = .{};
    box.init(open, 1 << 20);
    defer box.deinit();
    try t.expect(@sizeOf(why.Gone) <= 3 * 1024);
    for (0..why.Gone.SIZE + 1) |i| {
        const tokens = [_]u32{ @intCast(i + 1), 7 };
        box.s.gone.note(&tokens, 1, why.queryKey(&box.s, &tokens), .evicted);
    }
    try t.expect(box.s.gone.match(&box.s, &.{ 1, 7, 9 }, 0) == null); // the oldest was overwritten
    try t.expectEqual(why.Reason.evicted, box.s.gone.match(&box.s, &.{ 65, 7, 9 }, 0).?.reason);
    const next = box.s.gone.next;
    box.s.gone.note(&.{ 65, 7 }, 1, why.queryKey(&box.s, &.{ 65, 7 }), .shed);
    try t.expectEqual(next, box.s.gone.next);
    try t.expectEqual(why.Reason.shed, box.s.gone.match(&box.s, &.{ 65, 7, 9 }, 0).?.reason);
    try t.expect(box.s.gone.match(&box.s, &.{ 65, 7, 9 }, 1) == null); // only states past the resume point
}

test "prompt_cache why: diagnose reads the store and changes nothing a lookup or eviction reads" {
    var box: Box = .{};
    box.init(open, 1 << 20);
    defer box.deinit();
    try t.expect(box.keep(&words, 20, &.{}));
    try t.expect(box.keep(&words, 90, &.{}));
    try t.expect(box.keep(&others, 40, &.{}));
    const before = .{ box.s.counts, box.s.held, box.s.clock, box.s.entries.items.len, box.s.gone };
    var used: [8]u64 = undefined;
    for (box.s.entries.items, used[0..box.s.entries.items.len]) |e, *u| u.* = e.used;
    const changed = edited(30);
    for ([_][]const u32{ words[0..60], &changed, &words, &others }) |p| _ = why.diagnose(&box.s, p, &.{}, &.{}, box.s.find(p, &.{}, &.{}));
    try t.expectEqualDeep(before, .{ box.s.counts, box.s.held, box.s.clock, box.s.entries.items.len, box.s.gone });
    for (box.s.entries.items, used[0..box.s.entries.items.len]) |e, u| try t.expectEqual(u, e.used);
}
