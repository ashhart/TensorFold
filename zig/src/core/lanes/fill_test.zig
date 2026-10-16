//! Suffix-match grafts (lanes/fill.zig) on the fake target: a shared round that trims a grafted window stays exact.
const std = @import("std");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const win = @import("windows.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;

const gpa = std.testing.allocator;

/// The drafts the grafts follow: the head's, held by the target (no proposer, as the Metal CLI's streams), or copies.
const Trunk = enum { held, copied };

const Run = struct {
    replies: [][]u32,
    reasons: []sm.Reason,
    drafted: u64, // draft rows verified, all streams
    branch_rows: u64, // of those, grafted rows
    branch_kept: u64, // of those, rows on the accepted path

    fn deinit(r: Run) void {
        for (r.replies) |x| gpa.free(x);
        gpa.free(r.replies);
        gpa.free(r.reasons);
    }
};

fn run(trunk: Trunk, drafts: bool, n: usize, prompt: []const u32) !Run {
    // the cost of each row, not the 32-row cap, is what makes the allocator cut the held set's windows
    var costs: [16]Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    const head = drafts and trunk == .held;
    // a copied trunk shares 4 rows a round, so the allocator cuts into it
    const rows: u32 = if (trunk == .held) 32 else 4;
    var cfg = try Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = head, .speculate = head, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = rows, .max_streams = 8, .draft_streams = true }, 16, 15);
    defer cfg.deinit(gpa);
    cfg.fill_lanes = 9; // the Metal CLI's --lanes 8
    // the reply repeats from its first token, so suffix matches find branches; the head ranks its held chain
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = 0, .probe_prompt = prompt.len, .ranked = true };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, n);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, n);
    defer gpa.free(proposers);
    for (streams, proposers) |*s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        const proposer = if (trunk == .copied) p.proposer() else null;
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = prompt, .max_new = 120, .eos = &.{96}, .drafts = drafts, .proposer = proposer });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    var out: Run = .{ .replies = try gpa.alloc([]u32, n), .reasons = try gpa.alloc(sm.Reason, n), .drafted = 0, .branch_rows = 0, .branch_kept = 0 };
    for (out.replies, out.reasons, streams) |*r, *why, *s| {
        r.* = try gpa.dupe(u32, s.emitted());
        why.* = s.reason;
        out.drafted += s.drafted;
        out.branch_rows += s.branch_rows;
        out.branch_kept += s.branch_accepted;
    }
    return out;
}

test "a shared round that trims a window with suffix-match grafts replies as the one-token decode" {
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4 };
    for ([_]Trunk{ .held, .copied }) |trunk| {
        const want = try run(trunk, false, 1, &prompt);
        defer want.deinit();
        const got = try run(trunk, true, 2, &prompt);
        defer got.deinit();
        try std.testing.expect(got.branch_kept > 0 and got.branch_kept <= got.branch_rows and got.branch_rows <= got.drafted);
        for (got.replies, got.reasons) |r, why| {
            try std.testing.expectEqualSlices(u32, want.replies[0], r);
            try std.testing.expectEqual(want.reasons[0], why);
        }
    }
}

test "allocate keeps a grafted window's first drafts at every count, the trunk's before the grafts" {
    // flat costs, so a round fills its batch rows and the window's count is batch_rows - 1
    var costs: [16]Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 };
    const Window = struct { kind: win.Kind, held: u32, tokens: []const u32, parents: []const i32 };
    const windows = [_]Window{
        // three held drafts, then grafts: one after the last held draft, one off the first held draft and its child
        .{ .kind = .head, .held = 3, .tokens = &.{ 7, 8, 9 }, .parents = &.{ -1, 0, 1, 2, 0, 4 } },
        // three copied drafts, then grafts past their end: one after the last, its child and that child's sibling
        .{ .kind = .copy, .held = 0, .tokens = &.{ 4, 1, 5, 7, 8, 9 }, .parents = &.{ -1, 0, 1, 2, 3, 3 } },
    };
    for (windows) |t| for (1..1 + t.parents.len) |c| {
        var cfg = try Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = @intCast(c + 1), .max_streams = 8, .draft_streams = true }, 16, 15);
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer engine.deinit();
        var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &.{ 3, 1, 4 }, .max_new = 64, .drafts = true });
        defer s.deinit(gpa);
        var plans = [_]win.Plan{.{ .stream = &s, .position = 3, .kind = t.kind, .held = t.held, .tokens = t.tokens, .parents = t.parents, .branch_from = 4 }};
        try win.allocate(&engine, &plans);
        const p = plans[0];
        try std.testing.expectEqual(c, p.count());
        try std.testing.expectEqual(@min(t.held, c), p.held);
        try std.testing.expectEqualSlices(u32, t.tokens[0 .. c - @min(t.held, c)], p.tokens);
        try std.testing.expectEqualSlices(i32, t.parents[0..c], p.parents.?);
        try std.testing.expectEqual(@as(u32, if (c > 3) 4 else 0), p.branch_from);
    };
}
