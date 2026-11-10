//! Rounded cache sizes, warm work and slot 0's recurrent rewind across both kinds of client history.
const std = @import("std");
const api = @import("engine_api");
const pc = api.prompt_cache;
const tf = @import("tensorfold");
const lanes = tf.lanes;
const rules = @import("qwen27_resume.zig");
const Metal = @import("qwen27_lanes.zig").Metal;
const t = std.testing;
const a = t.allocator;

const Family = struct {
    at: u32 = 0,
    sum: u64 = 0,
    const State = struct { at: u32, sum: u64 };
    fn bytes(_: *anyopaque, at: u32) u64 {
        return tf.qwen27.snapshot.Spares.size(48 * (786432 * 4 + 30720 * 2) + @as(usize, at) * 65536);
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) !pc.Saved {
        const f: *Family = @ptrCast(@alignCast(ptr));
        try t.expectEqual(at, f.at);
        const x = try a.create(State);
        x.* = .{ .at = at, .sum = f.sum };
        return x;
    }
    fn restore(ptr: *anyopaque, _: ?*anyopaque, saved: pc.Saved) !void {
        const f: *Family = @ptrCast(@alignCast(ptr));
        const x: *State = @ptrCast(@alignCast(saved));
        f.* = .{ .at = x.at, .sum = x.sum };
    }
    fn drop(_: *anyopaque, saved: pc.Saved) void {
        a.destroy(@as(*State, @ptrCast(@alignCast(saved))));
    }
    fn snapshots(f: *Family) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
    fn advance(f: *Family, ids: []const u32, end: usize, store: ?*pc.Store, marks: []const u32, spans: []const [2]u32) void {
        while (f.at < end) {
            f.sum = f.sum *% 31 +% ids[f.at];
            f.at += 1;
            if (store) |s| if (std.mem.indexOfScalar(u32, marks, f.at) != null) {
                t.expect(s.keep(ids, f.at, null, &.{}, spans)) catch unreachable;
            };
        }
    }
};
fn fresh(ids: []const u32) u64 {
    var f: Family = .{};
    f.advance(ids, ids.len, null, &.{}, &.{});
    return f.sum;
}
fn no(_: *anyopaque, _: *const pc.Entry) bool {
    return false;
}
fn begin(store: *pc.Store, ids: []const u32, history: u32, rewind: u32, spans: []const [2]u32) !pc.Plan {
    store.reservation_guard = .{ .ctx = store, .needed = no, .exact = true };
    defer store.reservation_guard = null;
    return store.beginRewind(a, ids, history, rewind, &.{}, &.{}, null, spans);
}
fn never(_: *anyopaque) bool {
    return false;
}

fn turns(keepEnd: anytype, opening: anytype, kept: bool, budget: u64) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const alloc = arena.allocator();
    var f: Family = .{};
    var store = pc.Store.init(a, f.snapshots(), .{ .warm = .on, .chunk = 512, .chunk_marks = false }, budget);
    defer store.deinit();
    var standing: rules.Standing = .{};
    defer standing.deinit(a);
    const rewind = rules.needsRewind(budget, Family.bytes(&f, 65536));
    const lengths = if (kept) [_]usize{ 34083, 34361, 34639, 34831 } else [_]usize{ 34083, 34114, 34135, 34161 };
    const warm_lengths = if (kept) [_]usize{ 34097, 34368, 34666 } else [_]usize{ 34097, 34121, 34142 };
    const expected = if (kept) [_]u32{ 0, 34338, 34616, 34807 } else [_]u32{ 0, 34078, 34109, 34130 };
    var prompt = try alloc.alloc(u32, lengths[0]);
    for (prompt, 0..) |*id, i| id.* = @intCast(i + 1);
    var spans: []const [2]u32 = &.{};
    for (lengths, 0..) |n, turn| {
        const history: u32 = @intCast(n - 5);
        if (kept and turn > 0) {
            const other: [129]u32 = @splat(77);
            f = .{};
            f.advance(&other, 128, null, &.{}, &.{});
            try t.expect(store.keep(&other, 128, null, &.{}, &.{}));
            standing.rewind_at = 0;
            standing.leave(a, .{ .at = 128, .prompt_len = 128, .context_len = 129, .fill_at = 128, .ended = true }, &other, &.{});
        }
        const plan = try begin(&store, prompt, history, if (turn == 0) 45 else @intCast(n - 23), spans);
        defer a.free(plan.marks);
        const native = standing.plan(prompt, spans, plan.from);
        const from = switch (native) {
            .none => plan.from,
            .here, .back => |at| at,
        };
        errdefer std.debug.print("kept={any}, budget={d}, turn={d}, actual resume={d}, required={d}\n", .{ kept, budget, turn, from, expected[turn] });
        try t.expect(from >= expected[turn]);
        f = .{ .at = from, .sum = fresh(prompt[0..from]) };
        f.advance(prompt, n, &store, plan.marks, spans);
        try t.expectEqual(fresh(prompt), f.sum);
        const reply_len: usize = if (kept and turn == 2) 169 else 256;
        const context = try alloc.alloc(u32, n + reply_len);
        @memcpy(context[0..n], prompt);
        for (context[n..], 0..) |*id, i| id.* = @intCast(50000 + turn * 300 + i);
        const at: u32 = @intCast(context.len - 1);
        f.advance(context, at, null, &.{}, &.{});
        var stream = try lanes.Stream.init(a, .{ .id = "reply", .prompt = prompt, .max_new = @intCast(reply_len) });
        defer stream.deinit(a);
        stream.history_len = history;
        stream.decode_spans = spans;
        try stream.context.appendSlice(a, context[n..]);
        keepEnd(&store, a, &stream, at);
        standing.rewind_at = if (rewind) history else 0;
        standing.leave(a, .{ .at = at, .prompt_len = @intCast(n), .context_len = @intCast(context.len), .fill_at = @intCast(n), .ended = true }, context, spans);
        try t.expect(store.held <= budget);
        if (turn == 3) break;
        const warm = try alloc.alloc(u32, warm_lengths[turn]);
        @memcpy(warm[0 .. history + 4], context[0 .. history + 4]);
        for (warm[history + 4 ..], 0..) |*id, i| id.* = @intCast(60000 + turn * 100 + i);
        const wp = try begin(&store, warm, @intCast(warm.len), if (turn == 0) 45 else @intCast(n - 23), spans);
        defer a.free(wp.marks);
        var background = try lanes.Stream.init(a, .{ .id = "warm", .prompt = warm, .max_new = 0, .yield_check = .{ .ptr = &f, .check = never }, .reuse = .{ .marks = wp.marks } });
        defer background.deinit(a);
        background.decode_spans = spans;
        var offsets = [_]u32{at};
        var runner: tf.qwen27.decode_round.Runner = undefined;
        runner.capacity = 65536;
        runner.offsets = &offsets;
        runner.failed = false;
        runner.active = null;
        var back: Metal = undefined;
        back.runner = &runner;
        back.window = 16;
        back.slot_resume = true;
        back.vision = null;
        back.draft = null;
        back.standing = standing;
        if (rewind) {
            try t.expectError(error.Yielded, opening(&back, &background, 0));
            try t.expectEqual(at, runner.offsets[0]);
            try t.expectEqual(history, back.standing.rewind_at);
        } else {
            // Without the history rewind, the reasoning-free warm pass overwrites the standing reply.
            f = .{ .at = wp.from, .sum = fresh(warm[0..wp.from]) };
            f.advance(warm, warm.len, &store, wp.marks, spans);
            try t.expect(store.keepBorn(warm, @intCast(n), @intCast(warm.len), null, &.{}, spans));
            standing.leave(a, .{ .at = @intCast(warm.len), .prompt_len = @intCast(warm.len), .context_len = @intCast(warm.len), .fill_at = @intCast(warm.len), .ended = true }, warm, spans);
        }
        prompt = try alloc.alloc(u32, lengths[turn + 1]);
        const prefix = if (kept) context[0..at] else warm;
        @memcpy(prompt[0..prefix.len], prefix);
        for (prompt[prefix.len..], 0..) |*id, i| id.* = @intCast(70000 + turn * 100 + i);
        if (kept) spans = try api.cache_modes.reply(alloc, spans, @intCast(n), @intCast(context.len));
    }
}

fn priority(keepEnd: anytype, budget: u64, most: usize, both: bool) !void {
    var f: Family = .{};
    var store = pc.Store.init(a, f.snapshots(), .{ .warm = .on }, budget);
    defer store.deinit();
    store.most = most;
    const tokens: [34339]u32 = @splat(1);
    const prompt = tokens[0..34083];
    f.advance(&tokens, 34078, null, &.{}, &.{});
    try t.expect(store.keep(prompt, 34078, null, &.{}, &.{}));
    if (most >= 2) {
        const other: [129]u32 = @splat(77);
        const saved = f;
        f = .{};
        f.advance(&other, 128, null, &.{}, &.{});
        try t.expect(store.keep(&other, 128, null, &.{}, &.{}));
        f = saved;
    }
    var stream = try lanes.Stream.init(a, .{ .id = "caps", .prompt = prompt, .max_new = 256 });
    defer stream.deinit(a);
    stream.history_len = 34078;
    try stream.context.appendSlice(a, tokens[34083..]);
    f.advance(&tokens, 34338, null, &.{}, &.{});
    keepEnd(&store, a, &stream, 34338);
    const e = store.find(&tokens, &.{}, &.{.{ 34083, 34338 }}) orelse return error.ReplyEndMissing;
    try t.expectEqual(@as(u32, 34338), e.at);
    try t.expectEqual(both, store.find(prompt, &.{}, &.{}) != null);
    for (store.entries.items) |entry| try t.expectEqual(@as(u32, 0), entry.borrowers);
    try t.expect(store.held <= budget and store.entries.items.len <= most);
}

fn geometry() !void {
    var model: tf.qwen27.model.Model = undefined;
    model.config.head_dim = 256;
    model.config.kv_heads = 4;
    var runner: tf.qwen27.decode_round.Runner = undefined;
    var caches: [16]std.meta.Child(@TypeOf(runner.caches)) = undefined;
    runner.caches = &caches;
    runner.slots = 1;
    runner.model = &model;
    runner.gdn.budget.layers = 48;
    runner.gdn.budget.shape.state = 786432;
    runner.gdn.budget.shape.conv = 30720;
    var family: Family = .{};
    for ([_]u32{ 45, 7424, 34078, 34338, 65536 }) |at| {
        const actual = tf.qwen27.snapshot.Spares.size(tf.qwen27.snapshot.slotBytes(&runner, null, at));
        try t.expectEqual(actual, Family.bytes(&family, at));
    }
    const small = tf.qwen27.snapshot.rewindBytes(&runner, null);
    try t.expectEqual(@as(usize, 153944064), small);
    try t.expectEqual(@as(u64, 5184 << 20), Family.bytes(&family, 34078) + Family.bytes(&family, 34338));
    try t.expect(small + Family.bytes(&family, 34338) < 5 << 30);
}

fn shallow(keepEnd: anytype, cut: u32, borrow: bool, most: usize) !void {
    var f: Family = .{};
    var store = pc.Store.init(a, f.snapshots(), .{ .warm = .on }, 8 << 30);
    defer store.deinit();
    store.most = most;
    const tokens: [34339]u32 = @splat(1);
    const prompt = tokens[0..34083];
    f.advance(&tokens, cut, null, &.{}, &.{});
    try t.expect(store.keep(prompt, cut, null, &.{}, &.{}));
    if (cut == 45) store.entries.items[0].shared = true;
    const other: [129]u32 = @splat(77);
    f = .{};
    f.advance(&other, 128, null, &.{}, &.{});
    try t.expect(store.keep(&other, 128, null, &.{}, &.{}));
    const borrowed = store.entries.items[1];
    if (borrow) borrowed.borrowers += 1;
    defer if (borrow) {
        borrowed.borrowers -= 1;
    };
    var stream = try lanes.Stream.init(a, .{ .id = "borrowed", .prompt = prompt, .max_new = 256 });
    defer stream.deinit(a);
    stream.history_len = 34078;
    try stream.context.appendSlice(a, tokens[34083..]);
    f = .{};
    f.advance(&tokens, 34338, null, &.{}, &.{});
    keepEnd(&store, a, &stream, 34338);
    const e = store.find(&tokens, &.{}, &.{.{ 34083, 34338 }}) orelse return error.ReplyEndMissing;
    try t.expectEqual(@as(u32, 34338), e.at);
    if (borrow) try t.expectEqual(@as(u32, 1), borrowed.borrowers);
    try t.expect(store.find(&other, &.{}, &.{}) != null);
    if (cut == 34078) try t.expectEqual(most >= 3, store.find(prompt, &.{}, &.{}) != null);
}

pub fn exercise(keepEnd: anytype, opening: anytype) !void {
    try shallow(keepEnd, 45, false, 2);
    try shallow(keepEnd, 45, true, 2);
    try shallow(keepEnd, 34078, true, 2);
    try shallow(keepEnd, 34078, true, 3);
    try geometry();
    for ([_]u64{ 5 << 30, 8 << 30 }) |budget| for ([_]bool{ true, false }) |kept| try turns(keepEnd, opening, kept, budget);
    try priority(keepEnd, 5 << 30, 64, false);
    try priority(keepEnd, 8 << 30, 1, false);
    try priority(keepEnd, 8 << 30, 2, true);
    const end = rules.rewindEnd(33792, 34083, 34078);
    const bank = try tf.qwen27.prompt_marks.select(33792, @intCast(end), &.{34078});
    try t.expectEqual(@as(u32, 34078), bank.end);
    try t.expectEqual(@as(usize, 0), bank.count);
}
