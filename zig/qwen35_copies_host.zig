//! CUDA lane copy policies through the shared scheduler, without a driver, shader or checkpoint.
const std = @import("std");
const lanes = @import("lanes");
const qwen = @import("qwen35");
const a = std.testing.allocator;

fn facts() lanes.Model {
    if (comptime @hasDecl(qwen.native, "laneFacts")) return qwen.native.laneFacts(16, 32);
    return .{ .exact_width = 16, .hidden_rows = true, .batch_rows = 32, .max_streams = 32 };
}

fn configured() !lanes.Config {
    var m = facts();
    m.mtp = true;
    m.speculate = true;
    m.speculate_early = false;
    m.drafts = 7;
    return lanes.Config.init(a, m, 16, 15);
}

test "Suffix copies: CUDA facts enable five-token copies on the shared core" {
    try std.testing.expect(@hasDecl(qwen.native, "laneFacts"));
    var cfg = try configured();
    defer cfg.deinit(a);
    try std.testing.expectEqual(@as(i64, 5), cfg.enter_match);
    try std.testing.expect(cfg.lone_copies);
    try std.testing.expectEqual(@as(u32, 16), cfg.lone_cost_tile);
    try std.testing.expect(@hasField(lanes.Config, "context_copies"));
    if (comptime @hasField(lanes.Config, "context_copies")) try std.testing.expect(cfg.context_copies);
}

test "Suffix copies: the dense 27B copies from the root after 20 matching tokens beside DFlash2, after 5 alone" {
    const Shape = struct { hidden: usize, layers: usize, experts: usize };
    const dense27: Shape = .{ .hidden = 5120, .layers = 64, .experts = 0 };
    const moe: Shape = .{ .hidden = 2048, .layers = 40, .experts = 256 };
    const f27 = qwen.native.modelFacts(16, 32, dense27);
    try std.testing.expect(f27.root_copies);
    var plain = try lanes.Config.init(a, f27, 16, 15);
    defer plain.deinit(a);
    try std.testing.expectEqual(@as(i64, 5), plain.enter_match); // no drafter: copies are the only proposals
    try std.testing.expectEqual(@as(i64, lanes.windows.long_match), qwen.draft.copyMatch(dense27));
    const fm = qwen.native.modelFacts(16, 32, moe);
    try std.testing.expect(!fm.root_copies);
    try std.testing.expectEqual(@as(i64, 5), fm.copy_match);
    try std.testing.expectEqual(@as(i64, 0), qwen.draft.copyMatch(moe)); // 0: the target's
}

test "Suffix copies: draft tails never enter the suffix index and result storage may be the next tail" {
    try std.testing.expect(@hasDecl(lanes.SuffixLookup, "proposeAfter"));
    if (comptime @hasDecl(lanes.SuffixLookup, "proposeAfter")) {
        var p = try lanes.SuffixLookup.init(a, .{ .min_match = 5 });
        defer p.deinit();
        const history = [_]u32{ 7, 1, 2, 3, 4, 5, 6, 8, 9, 7, 1 };
        try std.testing.expectEqualSlices(u32, &.{ 5, 6, 8 }, try p.proposeAfter(&history, &.{ 2, 3, 4 }, 3));
        try std.testing.expectEqual(history.len, p.indexed);
        const own = try p.proposeAfter(&history, &.{ 2, 3, 4 }, 3);
        try std.testing.expectEqual(@as(usize, 0), (try p.proposeAfter(&history, own, 3)).len);
        try std.testing.expectEqual(history.len, p.indexed);
        try std.testing.expectEqualSlices(u32, &.{ 5, 6 }, try p.proposeAfter(&history, &.{ 2, 3, 4 }, 2));
        try std.testing.expectEqual(@as(usize, 0), (try p.proposeAfter(&history, &.{ 2, 3, 9 }, 3)).len);
        try std.testing.expectEqual(@as(i64, 0), p.last_match);
    }
}

test "Suffix copies: a lone copy shares a non-prefix draft path and appends within the current row cap" {
    var cfg = try configured();
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var proposer = try lanes.SuffixLookup.init(a, .{ .min_match = 5 });
    defer proposer.deinit();
    const history = [_]u32{ 11, 12, 13, 14, 15, 21, 22, 23, 77, 11, 12, 13, 14, 15 };
    var s = try lanes.Stream.init(a, .{ .id = "copy", .prompt = &history, .max_new = 24, .proposer = proposer.proposer() });
    defer s.deinit(a);
    s.pending = 15;
    s.next = .{ .count = 4, .tokens = try a.dupe(u32, &.{ 90, 21, 91, 22 }), .parents = try a.dupe(i32, &.{ -1, -1, 0, 1 }) };
    const p = try lanes.windows.plan(&engine, &s, null);
    try std.testing.expectEqual(lanes.windows.Kind.head, p.kind);
    try std.testing.expectEqualSlices(u32, &.{ 90, 21, 91, 22, 23, 77 }, p.tokens[0..6]);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, 0, 1, 3, 4 }, p.parents.?[0..6]);
    try std.testing.expect(p.count() + 1 <= 16);
    try std.testing.expectEqual(history.len, proposer.indexed);
}

const Options = struct { drafts: bool = true, copies: bool = true, restored: u32 = 0, streams: usize = 1, seed: ?u64 = null };
const Output = struct { tokens: [32]u32, copied: u64, accepted: u64, branches: u64, cached: u32 };
fn run(o: Options) !Output {
    var cfg = try configured();
    defer cfg.deinit(a);
    cfg.timed = false;
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const prompt = [_]u32{ 11, 12, 13, 14, 15, 21, 22, 23, 77, 11, 12, 13, 14 };
    var saved: std.ArrayList(u32) = .empty;
    defer saved.deinit(a);
    try saved.appendSlice(a, prompt[0..o.restored]);
    var streams: [8]lanes.Stream = undefined;
    var proposers: [8]lanes.SuffixLookup = undefined;
    var live: usize = 0;
    defer for (streams[0..live], proposers[0..live]) |*s, *p| {
        s.deinit(a);
        p.deinit();
    };
    for (streams[0..o.streams], proposers[0..o.streams], 0..) |*s, *p, i| {
        p.* = try lanes.SuffixLookup.init(a, .{ .min_match = 5 });
        s.* = try lanes.Stream.init(a, .{ .id = "cuda copies", .prompt = &prompt, .max_new = 32, .loop_guard = false, .drafts = o.drafts, .proposer = if (o.copies) p.proposer() else null, .sampling = if (o.seed) |seed| .{ .seed = seed + i, .temperature = 0.8 } else null, .reuse = .{ .saved = if (o.restored > 0) &saved else null, .at = o.restored } });
        live += 1;
        try engine.addStream(s);
    }
    while (engine.activeCount() > 0) engine.step() catch |err| {
        std.debug.print("copy case {any}, step {d}: {s}\n", .{ o, engine.steps, @errorName(err) });
        return err;
    };
    var out: Output = .{ .tokens = undefined, .copied = @intCast(proposers[0].proposed_tokens), .accepted = streams[0].accepted, .branches = streams[0].branch_rows, .cached = streams[0].cached };
    try std.testing.expectEqual(@as(usize, 32), streams[0].emitted().len);
    @memcpy(&out.tokens, streams[0].emitted());
    return out;
}

test "Suffix copies: drafted plain resumed fresh concurrent solo are identical with copies enabled and disabled" {
    for ([_]?u64{ null, 17, 91 }) |seed| {
        const plain = try run(.{ .drafts = false, .seed = seed });
        for ([_]bool{ false, true }) |copies| for ([_]u32{ 0, 5, 13 }) |restored| for ([_]usize{ 1, 2, 4, 8 }) |streams| {
            const got = try run(.{ .copies = copies, .restored = restored, .streams = streams, .seed = seed });
            try std.testing.expectEqual(plain.tokens, got.tokens);
            try std.testing.expectEqual(restored, got.cached);
            if (!copies) try std.testing.expectEqual(@as(u64, 0), got.copied);
            if (copies and seed == null and streams == 1) try std.testing.expect(got.copied > 0 and got.branches > 0);
        };
    }
}

test "Suffix copies: the history fake reads backend parents as complete window rows" {
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    const b = target.backend();
    var s = try lanes.Stream.init(a, .{ .id = "tree", .prompt = &.{ 1, 2, 3 }, .max_new = 16 });
    defer s.deinit(a);
    try b.prefill(&s);
    defer b.release(&s);
    var sampled: [3]u32 = undefined;
    var drafts: [2]u32 = undefined;
    const windows = [_]lanes.backend.Window{.{ .stream = &s, .pending = 4, .held = 0, .tokens = &.{ 9, 5 }, .parents = &.{ -1, 0, 0 }, .positions = &.{ 4, 5, 5 } }};
    var out = [_]lanes.backend.Verified{.{ .sampled = &sampled, .drafts = &drafts }};
    try b.verify(&windows, &out);
    try std.testing.expectEqual(lanes.fake.next(&.{ 1, 2, 3, 4, 5 }, null, 5), sampled[2]);
    try b.keep(&windows, &.{&.{ 0, 2 }});
    const h = try b.queue(&s, .{ .value = 6 }, 6);
    try std.testing.expectEqual(lanes.fake.next(&.{ 1, 2, 3, 4, 5, 6 }, null, 6), try b.read(h));
}

test "Suffix copies: the actual CUDA copy policy accepts five matching tokens and rejects four" {
    var cfg = try configured();
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    for ([_][]const u32{ &.{ 7, 1, 2, 3, 4, 5, 6, 8, 9, 7, 1, 2, 3, 4 }, &.{ 0, 1, 2, 3, 4, 5, 6, 8, 9, 7, 1, 2, 3, 4 } }, 0..) |history, i| {
        var p = try lanes.SuffixLookup.init(a, .{ .min_match = 4 });
        defer p.deinit();
        var s = try lanes.Stream.init(a, .{ .id = "match", .prompt = history, .max_new = 24, .proposer = p.proposer() });
        defer s.deinit(a);
        const copy = try lanes.windows.copyProposal(&engine, &s);
        if (i == 0) try std.testing.expect(copy.tokens.len > 0) else try std.testing.expectEqual(@as(usize, 0), copy.tokens.len);
    }
    var disabled = facts();
    disabled.copy_drafts = false;
    var off = try lanes.Config.init(a, disabled, 16, 15);
    defer off.deinit(a);
    try std.testing.expect(!off.context_copies and off.max_copy == 0);
}

fn slowCopy(history: []const u32, tail: []const u32, n: usize, min_match: usize, limit: usize) []const u32 {
    var best_end: usize = 0;
    var best: usize = 0;
    var end = history.len - 1;
    while (end >= n) : (end -= 1) {
        var length: usize = 0;
        while (length < @min(end, 64)) : (length += 1) {
            const j = history.len + tail.len - 1 - length;
            const right = if (j < history.len) history[j] else tail[j - history.len];
            if (history[end - 1 - length] != right) break;
        }
        if (length >= n and length > best) {
            best = length;
            best_end = end;
        }
    }
    return if (best < min_match) &.{} else history[best_end..@min(history.len, best_end + limit)];
}

test "Suffix copies: all suffix key widths agree with a slow history-only oracle across changing draft tails" {
    for (1..9) |n| for (0..20) |case| {
        var history: [32]u32 = undefined;
        for (&history, 0..) |*t, i| t.* = @intCast((i * 7 + case) % (case % 11 + 1));
        history[17 + case % 5] = 100;
        var p = try lanes.SuffixLookup.init(a, .{ .ngram = n, .min_match = @intCast(@max(n, 5)) });
        defer p.deinit();
        for (0..12) |length| {
            var tail: [12]u32 = undefined;
            for (tail[0..length], 0..) |*t, j| t.* = history[8 + j];
            for ([_]bool{ false, true }) |wrong| {
                if (wrong and length > 0) tail[length - 1] = 1000;
                const expected = slowCopy(&history, tail[0..length], n, @max(n, 5), 7);
                try std.testing.expectEqualSlices(u32, expected, try p.proposer().proposeAfter(&history, tail[0..length], 7));
                try std.testing.expectEqual(history.len, p.indexed);
            }
        }
    };
}
