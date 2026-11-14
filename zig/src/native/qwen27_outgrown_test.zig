//! Backend.grow as qwen27_slots.grow answers it, over slot_pages and a host pool: a slot out of pages ends its stream.
const std = @import("std");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const sp = tf.qwen27.slot_pages;
const kvp = tf.metal_kv_pages;
const t = std.testing;
const a = t.allocator;
const page = kvp.page_tokens;
const pass = 8; // the widest pass past a prompt that qwen27_slots.promiseRows adds, for this fake backend

var pages: ?*kvp.Pages = null;

/// Stream "a" is slot 0 and "b" slot 1; its rows so far and the window's, covered as the round's verify would.
fn grow(_: *anyopaque, w: lanes.backend.Window) bool {
    const p = pages orelse return true;
    const slot: usize = if (std.mem.eql(u8, w.stream.id, "a")) 0 else 1;
    sp.grow(p, slot, w.stream.prompt().len + w.stream.emitted().len + w.rows()) catch return false;
    return true;
}

const Run = struct {
    tokens: [2][]u32 = .{ &.{}, &.{} },
    reasons: [2]lanes.stream.Reason = .{ .none, .none },
    problems: [2][]const u8 = .{ "", "" },
    fn deinit(r: Run) void {
        for (r.tokens) |x| a.free(x);
    }
};

/// "a" (40 tokens, promised its prompt and a pass) and "b" (24, promised its whole reply), `count` from `first`.
fn run(with_pages: bool, first: usize, count: usize) !Run {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try lanes.Config.init(a, .{ .exact_width = 16, .gpu_tokens = true, .mtp = false, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true }, 16, 15);
    defer cfg.deinit(a);
    cfg.pipelined = false;
    cfg.timed = false;
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var vt = target.backend().vtable.*;
    if (with_pages) vt.grow = grow;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, .{ .ptr = &target, .vtable = &vt, .chunk_invariant = true }, clock.clock());
    defer e.deinit();
    var prompt: [240]u32 = undefined;
    for (&prompt, 0..) |*x, i| x.* = @intCast(3 + i % 50);
    const ids = [_][]const u8{ "a", "b" };
    const lens = [_]u32{ 40, 24 };
    const promised = [_]u64{ prompt.len + pass, prompt.len + 24 + pass };
    var streams: [2]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    for (streams[0..count], ids[first..][0..count], lens[first..][0..count], promised[first..][0..count], first..) |*s, id, n, rows, slot| {
        s.* = try lanes.Stream.init(a, .{ .id = id, .prompt = &prompt, .max_new = n });
        made += 1;
        if (pages) |p| if (with_pages) {
            try sp.promise(p, slot, rows); // qwen27_slots.bind as the stream opens its slot
            try sp.grow(p, slot, prompt.len); // its prompt pass
        };
        try e.addStream(s);
    }
    var guard: usize = 0;
    while (guard < 400) : (guard += 1) {
        var live = false;
        for (streams[0..count]) |s| live = live or !s.finished;
        if (!live) break;
        try e.step();
    }
    var r: Run = .{};
    for (streams[0..count], 0..) |s, i| {
        r.tokens[i] = try a.dupe(u32, s.emitted());
        r.reasons[i] = s.reason;
        r.problems[i] = s.problem;
    }
    return r;
}

test "qwen27 outgrown: the stream whose slot finds no page ends with length; the other gives its tokens alone; the pool is empty" {
    const l: tf.kv_pool.Layout = .{ .fixed_bytes = 0, .token_bytes = 64, .page_tokens = page, .snap_bytes = 0 };
    const p = try kvp.Pages.init(a, null, l, 2, 3, 2 * page, 0); // three pages: "a" one at its prompt, "b" two
    defer p.deinit();
    const alone = try run(false, 1, 1);
    defer alone.deinit();
    pages = p;
    defer pages = null;
    const cut = try run(true, 0, 2);
    defer cut.deinit();
    try t.expectEqual(lanes.stream.Reason.length, cut.reasons[0]);
    try t.expectEqualStrings("the server had no KV memory left for this reply's next rows", cut.problems[0]);
    try t.expect(cut.tokens[0].len < 40 and cut.tokens[0].len >= page - 240 - pass); // cut where its second page began
    try t.expectEqual(alone.reasons[0], cut.reasons[1]);
    try t.expectEqualSlices(u32, alone.tokens[0], cut.tokens[1]);
    for (0..2) |slot| sp.leave(p, slot, null); // each stream's unbind, as qwen27_slots.unbind releases its slot
    const c = p.pool.census();
    try t.expect(p.idle());
    try t.expectEqual(c.total, c.free);
    try t.expectEqual(@as(usize, 0), c.promised);
}
