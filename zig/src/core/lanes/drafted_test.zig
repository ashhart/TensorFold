//! An external drafter in front of the fake target: drafted == one-token rounds (greedy, sampled, shared), and the drafter
//! sees the rows it should (its drafts land, which a position or follow mistake would make chance).
const std = @import("std");
const Config = @import("config.zig").Config;
const Model = @import("config.zig").Model;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const be = @import("backend.zig");
const Drafter = @import("drafter.zig").Drafter;
const Drafted = @import("drafted.zig").Drafted;
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Sampling = @import("sampling.zig").Sampling;

const gpa = std.testing.allocator;

/// Rebuilds each stream's history from the target's features and drafts the fake target's next tokens, now and then wrong.
const FakeDrafter = struct {
    lanes: std.AutoHashMapUnmanaged(*sm.Stream, Lane) = .empty,
    absorbed: u64 = 0,
    first_start: ?u64 = null, // where the last opened stream's first absorb started

    const Lane = struct { history: std.ArrayList(u32) = .empty, pending: u32 = 0, held: std.ArrayList(u32) = .empty };
    const taps_ = [_]u32{ 44, 47, 49 };

    fn deinit(x: *FakeDrafter) void {
        var it = x.lanes.valueIterator();
        while (it.next()) |l| free(l);
        x.lanes.deinit(gpa);
    }

    fn free(l: *Lane) void {
        l.history.deinit(gpa);
        l.held.deinit(gpa);
    }

    fn drafter(x: *FakeDrafter) Drafter {
        return .{ .ptr = x, .vtable = &.{ .taps = taps, .depth = depth, .open = open, .absorb = absorb, .hold = hold, .held = held, .release = release } };
    }

    fn self(ptr: *anyopaque) *FakeDrafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn taps(_: *anyopaque) []const u32 {
        return &taps_;
    }

    fn depth(_: *anyopaque) u32 {
        return 6;
    }

    fn open(ptr: *anyopaque, s: *sm.Stream) anyerror!void {
        const x = self(ptr);
        const got = try x.lanes.getOrPut(gpa, s);
        if (got.found_existing) free(got.value_ptr);
        got.value_ptr.* = .{};
        x.first_start = null;
    }

    fn absorb(ptr: *anyopaque, s: *sm.Stream, f: be.Features, start: u64, follow: []const u32) anyerror!void {
        const x = self(ptr);
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        if (start > l.history.items.len and l.history.items.len > 0) return error.Gap;
        if (f.row_bytes != 4 or follow.len != f.rows) return error.Shape;
        const rows: [*]const u32 = @ptrFromInt(f.buffer + f.offset);
        if (l.history.items.len == 0 and x.first_start == null) x.first_start = start;
        while (l.history.items.len < start) try l.history.append(gpa, fake.vocab + 1); // restored rows: never seen
        l.history.shrinkRetainingCapacity(start);
        try l.history.appendSlice(gpa, rows[0..f.rows]);
        // the token after each row is the next row's
        for (follow[0 .. follow.len - 1], rows[1..f.rows]) |a, b| if (a != b) return error.FollowMismatch;
        l.pending = follow[follow.len - 1];
        x.absorbed += f.rows;
    }

    fn hold(ptr: *anyopaque, s: *sm.Stream, position: u64, n: u32) anyerror!void {
        const l = self(ptr).lanes.getPtr(s) orelse return error.UnknownStream;
        if (position != l.history.items.len + 1) return error.PositionMismatch;
        var guess: std.ArrayList(u32) = .empty;
        defer guess.deinit(gpa);
        try guess.appendSlice(gpa, l.history.items);
        try guess.append(gpa, l.pending);
        l.held.clearRetainingCapacity();
        for (0..n) |j| {
            const at = position + j;
            var t = fake.next(guess.items, s.sampling, at);
            if ((at *% 2654435761 + j) % 5 == 0) t = (t + 1) % fake.vocab;
            try l.held.append(gpa, t);
            try guess.append(gpa, t);
        }
    }

    fn held(ptr: *anyopaque, s: *sm.Stream, out: []u32) anyerror!usize {
        const l = self(ptr).lanes.getPtr(s) orelse return error.UnknownStream;
        const n = @min(out.len, l.held.items.len);
        @memcpy(out[0..n], l.held.items[0..n]);
        return n;
    }

    fn release(ptr: *anyopaque, s: *sm.Stream) void {
        const x = self(ptr);
        if (x.lanes.fetchRemove(s)) |kv| {
            var l = kv.value;
            free(&l);
        }
    }
};

const Case = struct { prompt: []const u32, max_new: u32 = 40, sampling: ?Sampling = null, drafts: bool = true, restored: u32 = 0 };
const Out = struct { tokens: [][]u32, drafted: u64, accepted: u64, absorbed: u64, first_start: ?u64, cached: u32 };

fn run(cases: []const Case, external: bool) !Out {
    var costs: [16]Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    const base: Model = .{ .exact_width = 16, .gpu_tokens = true, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true };
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var head: FakeDrafter = .{};
    defer head.deinit();
    var wrapped: Drafted = .{ .gpa = gpa, .target = target.backend(), .drafter = head.drafter() };
    defer wrapped.deinit();
    var cfg = try Config.init(gpa, if (external) wrapped.facts(base) else base, 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, if (external) wrapped.backend() else target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    const kept = try gpa.alloc(?*anyopaque, cases.len);
    defer gpa.free(kept);
    defer for (kept) |k| if (k) |saved| target.drop(saved);
    for (cases, streams, proposers, kept) |c, *s, *p, *k| {
        k.* = null;
        if (c.restored > 0) { // the prompt's first `restored` tokens as a kept state, saved by an earlier pass
            var early = try sm.Stream.init(gpa, .{ .id = "early", .prompt = c.prompt[0..c.restored], .max_new = 1, .eos = &.{96} });
            defer early.deinit(gpa);
            const b = target.backend();
            try b.prefill(&early);
            k.* = try target.save(&early, c.restored);
            b.release(&early);
        }
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer(), .reuse = .{ .saved = k.*, .at = c.restored } });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    var out: Out = .{ .tokens = try gpa.alloc([]u32, cases.len), .drafted = 0, .accepted = 0, .absorbed = head.absorbed, .first_start = head.first_start, .cached = streams[0].cached };
    for (out.tokens, streams) |*o, *s| {
        o.* = try gpa.dupe(u32, s.emitted());
        out.drafted += s.drafted;
        out.accepted += s.accepted;
    }
    return out;
}

fn freeOut(o: Out) void {
    for (o.tokens) |t| gpa.free(t);
    gpa.free(o.tokens);
}

const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9 };

test "an external drafter's rounds commit the one-token decode, greedy and sampled" {
    for ([_]?Sampling{ null, .{ .seed = 5, .temperature = 0.7, .top_k = 0, .top_p = 0.95 } }) |s| {
        const drafted = try run(&.{.{ .prompt = &p1, .sampling = s }}, true);
        defer freeOut(drafted);
        const plain = try run(&.{.{ .prompt = &p1, .sampling = s, .drafts = false }}, false);
        defer freeOut(plain);
        try std.testing.expectEqualSlices(u32, plain.tokens[0], drafted.tokens[0]);
        // the drafter saw the right rows: four in five of its drafts are the target's, so far more than chance (1 in 97) land
        try std.testing.expect(drafted.drafted > 0 and drafted.accepted * 3 > drafted.drafted);
        try std.testing.expect(drafted.absorbed >= p1.len - 1 + drafted.tokens[0].len - 1);
    }
}

test "an external drafter's shared rounds commit what each stream commits alone" {
    const sampled: Sampling = .{ .seed = 9, .temperature = 1.0, .top_k = 0, .top_p = 0.9 };
    const together = try run(&.{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = sampled, .max_new = 30 } }, true);
    defer freeOut(together);
    const one = try run(&.{.{ .prompt = &p1, .drafts = false }}, false);
    defer freeOut(one);
    const two = try run(&.{.{ .prompt = &p2, .sampling = sampled, .max_new = 30, .drafts = false }}, false);
    defer freeOut(two);
    try std.testing.expectEqualSlices(u32, one.tokens[0], together.tokens[0]);
    try std.testing.expectEqualSlices(u32, two.tokens[0], together.tokens[1]);
}

test "a target without features cannot take an external drafter" {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var b = target.backend();
    var vt = b.vtable.*;
    vt.features = null;
    b.vtable = &vt;
    var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &p1, .max_new = 4, .eos = &.{96} });
    defer s.deinit(gpa);
    try b.prefill(&s);
    defer b.release(&s);
    try std.testing.expectError(error.NoFeatures, b.features(&s, &.{49}, 0, 1));
}

test "a prompt restored from a kept state: the drafter starts at the restored row, the output is the one-token decode" {
    const long = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4 };
    for ([_]?Sampling{ null, .{ .seed = 3, .temperature = 0.8, .top_k = 0, .top_p = 0.95 } }) |s| {
        const drafted = try run(&.{.{ .prompt = &long, .sampling = s, .restored = 12 }}, true);
        defer freeOut(drafted);
        const plain = try run(&.{.{ .prompt = &long, .sampling = s, .drafts = false }}, false);
        defer freeOut(plain);
        try std.testing.expectEqual(@as(u32, 12), drafted.cached);
        try std.testing.expectEqual(@as(?u64, 12), drafted.first_start);
        try std.testing.expectEqualSlices(u32, plain.tokens[0], drafted.tokens[0]);
    }
}
