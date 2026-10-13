//! Recall in the round loop on the fake target: copies and grafts from a populated record change no token.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const recall = @import("recall.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Sampling = @import("sampling.zig").Sampling;

const gpa = std.testing.allocator;

const Case = struct { prompt: []const u32, max_new: u32 = 120, sampling: ?Sampling = null, drafts: bool = true };

const Opts = struct { lone_copies: bool = true, fill_lanes: u32 = 0, own_copies: bool = true, copies: bool = true };

const Out = struct {
    emitted: [][]u32,
    rounds: u64 = 0,
    proposals: u64 = 0, // Recall's copies offered, and their ids landed
    accepted: u64 = 0,
    recall_kept: u64 = 0, // graft rows kept, all sources

    fn deinit(o: Out) void {
        for (o.emitted) |e| gpa.free(e);
        gpa.free(o.emitted);
    }
};

fn model(o: Opts) !Config {
    var costs: [16]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try Config.init(gpa, .{
        .exact_width = 16,
        .gpu_tokens = true,
        .mtp = true,
        .speculate = true,
        .speculate_early = false,
        .drafts = 4,
        .window_costs = &costs,
        .mtp_step_ms = 0.5,
        .hidden_rows = true,
        .batch_rows = 32,
        .max_streams = 8,
        .draft_streams = true,
        .lone_copies = o.lone_copies,
    }, 16, 15);
    cfg.fill_lanes = o.fill_lanes;
    if (!o.copies) cfg.max_copy = 0; // no copy windows: the head's chain every round, grafts beside it
    return cfg;
}

/// Every case admitted together and stepped until done, each stream reading `shared` when given.
fn run(cases: []const Case, shared: ?*recall.Shared, o: Opts) !Out {
    var cfg = try model(o);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .ranked = o.fill_lanes > 0 };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    const readers = try gpa.alloc(recall.Reader, cases.len);
    defer gpa.free(readers);
    for (cases, streams, proposers, readers) |c, *s, *p, *r| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        r.* = .{ .gpa = gpa, .shared = shared orelse undefined };
        s.* = try sm.Stream.init(gpa, .{
            .id = "s",
            .prompt = c.prompt,
            .max_new = c.max_new,
            .eos = &.{96},
            .sampling = c.sampling,
            .drafts = c.drafts,
            .proposer = if (o.own_copies) p.proposer() else null,
            .recall = if (shared != null) r else null,
        });
    }
    defer for (streams, proposers, readers) |*s, *p, *r| {
        s.deinit(gpa);
        p.deinit();
        r.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    var out: Out = .{ .emitted = try gpa.alloc([]u32, cases.len) };
    for (out.emitted, streams, readers) |*e, *s, r| {
        e.* = try gpa.dupe(u32, s.emitted());
        out.rounds += s.rounds;
        out.proposals += r.proposals;
        out.accepted += r.accepted;
        out.recall_kept += s.branch_accepted;
    }
    return out;
}

fn prompt(comptime n: usize, seed: u32) [n]u32 {
    var out: [n]u32 = undefined;
    var x: u32 = seed;
    for (&out) |*t| {
        x = x *% 1103515245 +% 12345;
        t.* = (x >> 16) % 90; // under the eos id 96
    }
    return out;
}

/// The record as the server writes it: the prompt, then its last 32 ids with the reply.
fn remember(shared: *recall.Shared, p: []const u32, reply: []const u32) !void {
    try shared.record.add(p);
    var seg: std.ArrayList(u32) = .empty;
    defer seg.deinit(gpa);
    try seg.appendSlice(gpa, p[p.len -| 32..]);
    try seg.appendSlice(gpa, reply);
    try shared.record.add(seg.items);
}

fn newShared() !recall.Shared {
    return .{ .io = std.testing.io, .record = try recall.Record.init(gpa, 16) };
}

test "a repeated request copies its earlier reply from the record: the same tokens, fewer rounds, greedy and sampled" {
    const p = prompt(40, 7);
    for ([_]?Sampling{ null, .{ .seed = 99, .temperature = 0.8 } }) |sampling| {
        const plain = try run(&.{.{ .prompt = &p, .sampling = sampling, .drafts = false }}, null, .{});
        defer plain.deinit();
        const before = try run(&.{.{ .prompt = &p, .sampling = sampling }}, null, .{});
        defer before.deinit();
        try std.testing.expectEqualSlices(u32, plain.emitted[0], before.emitted[0]);
        var shared = try newShared();
        defer shared.record.deinit();
        try remember(&shared, &p, plain.emitted[0]);
        const with = try run(&.{.{ .prompt = &p, .sampling = sampling }}, &shared, .{});
        defer with.deinit();
        try std.testing.expectEqualSlices(u32, plain.emitted[0], with.emitted[0]);
        try std.testing.expect(with.accepted > 0 and with.rounds < before.rounds);
    }
}

test "a lone stream whose family keeps context copies off still takes Recall's when they pay, exactly" {
    const p = prompt(48, 3);
    const plain = try run(&.{.{ .prompt = &p, .drafts = false }}, null, .{ .lone_copies = false });
    defer plain.deinit();
    const before = try run(&.{.{ .prompt = &p }}, null, .{ .lone_copies = false });
    defer before.deinit();
    var shared = try newShared();
    defer shared.record.deinit();
    try remember(&shared, &p, plain.emitted[0]);
    const with = try run(&.{.{ .prompt = &p }}, &shared, .{ .lone_copies = false });
    defer with.deinit();
    try std.testing.expectEqualSlices(u32, plain.emitted[0], with.emitted[0]);
    try std.testing.expect(with.accepted > 0 and with.rounds < before.rounds);
}

test "concurrent streams reading one record each equal their one-token reply" {
    const ps = [_][40]u32{ prompt(40, 11), prompt(40, 12), prompt(40, 13) };
    var shared = try newShared();
    defer shared.record.deinit();
    var plains: [3][]u32 = undefined;
    var outs: [3]Out = undefined;
    for (&ps, &plains, &outs) |*p, *x, *o| {
        o.* = try run(&.{.{ .prompt = p, .drafts = false }}, null, .{});
        x.* = o.emitted[0];
        try remember(&shared, p, x.*);
    }
    defer for (outs) |o| o.deinit();
    const together = try run(&.{ .{ .prompt = &ps[0] }, .{ .prompt = &ps[1] }, .{ .prompt = &ps[2] } }, &shared, .{});
    defer together.deinit();
    for (plains, together.emitted) |want, got| try std.testing.expectEqualSlices(u32, want, got);
    try std.testing.expect(together.accepted > 0);
}

test "a hostile record of near-miss continuations changes no token, and its bands back off" {
    const p = prompt(40, 21);
    const plain = try run(&.{.{ .prompt = &p, .drafts = false }}, null, .{});
    defer plain.deinit();
    const reply = plain.emitted[0];
    var shared = try newShared();
    defer shared.record.deinit();
    var seg: std.ArrayList(u32) = .empty;
    defer seg.deinit(gpa);
    var j: usize = reply.len - 12;
    while (j > 0) { // the true text to j, a wrong id, then the truth: the newest match always misses at once
        j -= 1;
        seg.clearRetainingCapacity();
        try seg.appendSlice(gpa, &p);
        try seg.appendSlice(gpa, reply[0..j]);
        try seg.append(gpa, (reply[j] + 1) % 90);
        try seg.appendSlice(gpa, reply[j + 1 .. j + 12]);
        try shared.record.add(seg.items);
    }
    const alone: Opts = .{ .own_copies = false }; // Recall's copies only, so every one of them meets the hostile record
    const before = try run(&.{.{ .prompt = &p }}, null, alone);
    defer before.deinit();
    const with = try run(&.{.{ .prompt = &p }}, &shared, alone);
    defer with.deinit();
    try std.testing.expectEqualSlices(u32, reply, with.emitted[0]);
    try std.testing.expectEqual(@as(u64, 0), with.accepted);
    try std.testing.expect(with.proposals <= 6 and with.rounds <= before.rounds + 2); // a few misses, then priced out
    try std.testing.expect(shared.rates.rate(recall.bands.len - 1) < 0.7); // its matches are all 64 ids long
    const again = try run(&.{.{ .prompt = &p }}, &shared, alone); // a later request meets the backed-off band
    defer again.deinit();
    try std.testing.expectEqualSlices(u32, reply, again.emitted[0]);
    try std.testing.expect(again.proposals < with.proposals and again.rounds <= before.rounds + 1);
}

test "Recall's continuations graft beside the head's chain and keep the reply's tokens" {
    const p = prompt(40, 31);
    const o: Opts = .{ .fill_lanes = 12, .copies = false };
    const plain = try run(&.{.{ .prompt = &p, .drafts = false }}, null, o);
    defer plain.deinit();
    const before = try run(&.{.{ .prompt = &p }}, null, o);
    defer before.deinit();
    try std.testing.expectEqualSlices(u32, plain.emitted[0], before.emitted[0]);
    var shared = try newShared();
    defer shared.record.deinit();
    try remember(&shared, &p, plain.emitted[0]);
    const with = try run(&.{.{ .prompt = &p }}, &shared, o);
    defer with.deinit();
    try std.testing.expectEqualSlices(u32, plain.emitted[0], with.emitted[0]);
    try std.testing.expect(with.recall_kept > before.recall_kept and with.rounds < before.rounds);
}
