//! Host tests of the conformance ledger (docs/conformance-ledger.md): one output and cache oracle, research rank 0.
const std = @import("std");
const accept = @import("accept.zig");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const win = @import("windows.zig");
const Sampling = @import("sampling.zig").Sampling;
const SuffixLookup = @import("proposer.zig").SuffixLookup;

const gpa = std.testing.allocator;
const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4 };
const keyed: Sampling = .{ .seed = 23, .temperature = 1.0, .top_k = 0, .top_p = 0.9 };

/// How the engine drafts: one-token rounds, the head's chains, or suffix-match trees beside the head (fill.zig).
const Shape = enum { plain, chain, tree };
const Opts = struct { shape: Shape = .plain, repeat: bool = false, sampling: ?Sampling = null, resumed: bool = false, max_new: u32 = 48, eos: []const u32 = &.{96} };

const Run = struct {
    tokens: []u32,
    reason: sm.Reason,
    steps: usize,
    digests: []u64, // the cache's digest after every step
    drafted: u64,
    branch_rows: u64,
    cached: u32,

    fn deinit(r: Run) void {
        gpa.free(r.tokens);
        gpa.free(r.digests);
    }
};

/// One stream to its end, its cache after every step the fresh decode's (prompt, committed, pending fed or not).
fn run(o: Opts) !Run {
    var saved: std.ArrayList(u32) = .empty; // a kept state after the prompt's first 5 tokens (fake.zig's seed)
    defer saved.deinit(gpa);
    try saved.appendSlice(gpa, prompt[0..5]);
    var costs: [16]Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    const head = o.shape != .plain;
    var cfg = try Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = head, .speculate = head, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true }, 16, 15);
    defer cfg.deinit(gpa);
    if (o.shape == .tree) cfg.fill_lanes = 9; // grafts beside the head's chain, as fill_test
    // trees need a reply that repeats (suffix matches) and a head that ranks its chain; `repeat` alone: fake.next's
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = if (o.repeat) 0 else null, .probe_prompt = prompt.len, .ranked = o.shape == .tree };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var lookup = try SuffixLookup.init(gpa, .{ .min_match = 1000 }); // no copies: every draft is the head's or a graft
    defer lookup.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "c", .prompt = &prompt, .max_new = o.max_new, .eos = o.eos, .sampling = o.sampling, .drafts = head, .proposer = lookup.proposer(), .reuse = if (o.resumed) .{ .saved = &saved, .at = 5 } else .{} });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    var digests: std.ArrayList(u64) = .empty;
    errdefer digests.deinit(gpa);
    var steps: usize = 0;
    while (engine.activeCount() > 0) : (steps += 1) {
        try engine.step();
        const rows = target.cache(&s) orelse continue; // released at its end
        const fresh = s.context.items;
        try std.testing.expect(rows.len == fresh.len or rows.len + 1 == fresh.len);
        try std.testing.expectEqual(fake.Fake.digest(fresh[0..rows.len]), fake.Fake.digest(rows));
        try digests.append(gpa, fake.Fake.digest(rows));
    }
    return .{ .tokens = try gpa.dupe(u32, s.emitted()), .reason = s.reason, .steps = steps, .digests = try digests.toOwnedSlice(gpa), .drafted = s.drafted, .branch_rows = s.branch_rows, .cached = s.cached };
}

/// fake.next's one-token decode after the prompt: the reference every drafted shape must equal.
fn decode(o: Opts) ![]u32 {
    var h: std.ArrayList(u32) = .empty;
    defer h.deinit(gpa);
    try h.appendSlice(gpa, &prompt);
    while (h.items.len - prompt.len < o.max_new) {
        const t = fake.next(h.items, o.sampling, h.items.len);
        try h.append(gpa, t);
        if (std.mem.indexOfScalar(u32, o.eos, t) != null) break;
    }
    return gpa.dupe(u32, h.items[prompt.len..]);
}

test "conformance: fresh and resumed serial decode emit the same tokens and hold the same cache every round (C1, C7)" {
    for ([_]Shape{ .plain, .chain }) |shape| for ([_]?Sampling{ null, keyed }) |sampling| {
        const fresh = try run(.{ .shape = shape, .sampling = sampling });
        defer fresh.deinit();
        const resumed = try run(.{ .shape = shape, .sampling = sampling, .resumed = true });
        defer resumed.deinit();
        try std.testing.expectEqual(@as(u32, 0), fresh.cached);
        try std.testing.expectEqual(@as(u32, 5), resumed.cached);
        try std.testing.expectEqualSlices(u32, fresh.tokens, resumed.tokens);
        try std.testing.expectEqual(fresh.reason, resumed.reason);
        // one-token rounds: the resumed cache is the fresh one at every step, not only at the end
        if (shape == .plain) try std.testing.expectEqualSlices(u64, fresh.digests, resumed.digests);
        const want = try decode(.{ .sampling = sampling });
        defer gpa.free(want);
        try std.testing.expectEqualSlices(u32, want, fresh.tokens);
    };
}

test "conformance: tree windows commit the plain decode's tokens, and their rejected rows never stay in the cache (C2, C7)" {
    // one repeating reply for every shape, so suffix matches graft branches beside the head's chain
    const plain = try run(.{ .shape = .plain, .repeat = true });
    defer plain.deinit();
    for ([_]Shape{ .chain, .tree }) |shape| for ([_]bool{ false, true }) |resumed| {
        const got = try run(.{ .shape = shape, .repeat = true, .resumed = resumed });
        defer got.deinit();
        try std.testing.expectEqualSlices(u32, plain.tokens, got.tokens);
        try std.testing.expectEqual(plain.reason, got.reason);
        try std.testing.expect(got.drafted > 0);
        if (shape == .tree) try std.testing.expect(got.branch_rows > 0); // tree rows were verified, not chains alone
    };
}

test "conformance: the same request twice, greedy and keyed, emits the same tokens in the same rounds (C3)" {
    for ([_]Shape{ .plain, .chain, .tree }) |shape| for ([_]?Sampling{ null, keyed }) |sampling| {
        const a = try run(.{ .shape = shape, .repeat = shape == .tree, .sampling = sampling });
        defer a.deinit();
        const b = try run(.{ .shape = shape, .repeat = shape == .tree, .sampling = sampling });
        defer b.deinit();
        try std.testing.expectEqualSlices(u32, a.tokens, b.tokens);
        try std.testing.expectEqual(a.steps, b.steps);
        try std.testing.expectEqual(a.drafted, b.drafted);
        try std.testing.expectEqualSlices(u64, a.digests, b.digests);
    };
    // the key matters: a keyed reply is not the greedy one (else the keyed rows above pin nothing)
    const greedy = try decode(.{});
    defer gpa.free(greedy);
    const sampled = try decode(.{ .sampling = keyed });
    defer gpa.free(sampled);
    try std.testing.expect(!std.mem.eql(u32, greedy, sampled));
}

test "conformance: eos and max_new end a drafted reply where the plain decode ends, with its reason (C4)" {
    for ([_]?Sampling{ null, keyed }) |sampling| {
        const long = try decode(.{ .sampling = sampling, .max_new = 48, .eos = &.{} });
        defer gpa.free(long);
        // an eos the plain decode reaches part way: the first token not seen before it
        var at: usize = 3;
        while (std.mem.indexOfScalar(u32, long[0..at], long[at]) != null) at += 1;
        const eos = [_]u32{long[at]};
        for ([_]Shape{ .plain, .chain }) |shape| {
            const stopped = try run(.{ .shape = shape, .sampling = sampling, .eos = &eos });
            defer stopped.deinit();
            try std.testing.expectEqual(sm.Reason.stop, stopped.reason);
            try std.testing.expectEqualSlices(u32, long[0 .. at + 1], stopped.tokens);
            const capped = try run(.{ .shape = shape, .sampling = sampling, .max_new = @intCast(at), .eos = &.{} });
            defer capped.deinit();
            try std.testing.expectEqual(sm.Reason.length, capped.reason);
            try std.testing.expectEqualSlices(u32, long[0..at], capped.tokens);
        }
    }
}

test "conformance: a parent at or after its child is refused at accept.depths and windows.build, before any backend (C6)" {
    try std.testing.expectError(error.ParentAfterChild, accept.depths(gpa, &.{ -1, 0, 2 })); // its own parent
    try std.testing.expectError(error.ParentAfterChild, accept.depths(gpa, &.{ -1, 2, 0 })); // a later row's child
    const ds = try accept.depths(gpa, &.{ -1, 0, 0, 1 });
    defer gpa.free(ds);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 1, 2 }, ds);
    var cfg = try Config.init(gpa, .{ .exact_width = 16, .mtp = true, .speculate = true, .drafts = 4 }, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "p", .prompt = &prompt, .max_new = 8 });
    defer s.deinit(gpa);
    s.pending = 7;
    // drafts' parents (-1: the pending row): draft 0 under draft 1, a parent after its child
    const bad: win.Plan = .{ .stream = &s, .position = prompt.len, .kind = .head, .tokens = &.{ 11, 12 }, .parents = &.{ 1, -1 } };
    try std.testing.expectError(error.ParentAfterChild, win.build(&engine, bad, false));
    const good: win.Plan = .{ .stream = &s, .position = prompt.len, .kind = .head, .tokens = &.{ 11, 12, 13 }, .parents = &.{ -1, -1, 0 } };
    const w = try win.build(&engine, good, false);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 0, 1 }, w.parents.?);
    try std.testing.expectEqualSlices(u64, &.{ prompt.len + 1, prompt.len + 2, prompt.len + 2, prompt.len + 3 }, w.positions);
    try std.testing.expectEqual(@as(u64, 0), target.rounds); // nothing reached the backend
}
