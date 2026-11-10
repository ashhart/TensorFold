//! A stream count's first plain trial pays one-off costs: the plain price comes from later trials alone.
const std = @import("std");
const t = std.testing;
const a = t.allocator;
const be = @import("backend.zig");
const fake = @import("fake.zig");
const Stream = @import("stream.zig").Stream;
const Engine = @import("engine.zig").Engine;
const Config = @import("config.zig").Config;

const plain_ms = 20.0; // a warm all-plain round of four streams
const warmup_ms = 4.0; // what the first all-plain round of four streams pays once
const forward_ms = 17.0; // the load-time table's four-row forward
const later_from = 200; // four-stream rounds from which every round costs `Case.later` more
const settled_from = 129; // the second trial's round and the one after it, which holds no drafts

const Case = struct {
    accept: f64, // the share of drafts the target keeps, spread evenly, and the streams' prior
    row_ms: f64, // what a drafted row adds to a round
    table_row_ms: f64, // what the load-time table charges a row past the fourth
    later: f64 = 0,
    max_new: u32,
};

const Target = struct {
    target: fake.Fake = .{ .gpa = a },
    case: Case,
    cold: bool = true,
    ms: f64 = 0,
    windows: usize = 0,
    fours: u64 = 0, // shared rounds of four streams
    credit: f64 = 0,

    fn of(ptr: *anyopaque) *Target {
        const x: *fake.Fake = @ptrCast(@alignCast(ptr));
        return @fieldParentPtr("target", x);
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const h = of(ptr);
        try h.target.backend().vtable.verify(ptr, windows, out);
        h.windows = windows.len;
        h.fours += @intFromBool(windows.len == 4);
        const extra: f64 = @floatFromInt(h.target.last_rows - windows.len);
        h.ms = plain_ms * @as(f64, @floatFromInt(windows.len)) / 4 + h.case.row_ms * extra;
        if (h.fours >= later_from) h.ms += h.case.later;
        // the streams' first round switches their prompts to decoding; the one-off cost lands on the next plain one
        if (extra == 0 and windows.len == 4 and h.cold and windows[0].stream.rounds > 0) {
            h.ms += warmup_ms;
            h.cold = false;
        }
    }

    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const h = of(ptr);
        for (requests) |r| {
            if (r.depth > 0) h.credit += h.case.accept;
            const kept = h.credit >= 1;
            if (kept) h.credit -= 1;
            h.target.head_accepts = @intFromBool(kept);
            try h.target.backend().vtable.draft(ptr, &[_]be.DraftRequest{r});
        }
    }

    fn start(_: *anyopaque) void {}

    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        const h: *Target = @ptrCast(@alignCast(ptr));
        return h.ms;
    }
};

const Run = struct {
    tokens: [4][]u32,
    trials: [8]u64 = @splat(0), // the four-stream round of each plain trial
    count: usize = 0,
    rounds_after: usize = 0, // four-stream rounds past `settled_from` but trials, and those that drafted
    drafted_after: usize = 0,

    fn deinit(r: *Run) void {
        for (r.tokens) |s| a.free(s);
    }
};

/// Four streams on a fake target priced by `case`; with `drafts`, each trial's plain price is checked as it lands.
fn run(case: Case, sampled: bool, drafts: bool) !Run {
    const costs = [_]@import("config.zig").Cost{ .{ .width = 1, .ms = 10 }, .{ .width = 4, .ms = forward_ms }, .{ .width = 8, .ms = forward_ms + 4 * case.table_row_ms } };
    // a slow estimate keeps each stream's acceptance near the target's own share
    var cfg = try Config.init(a, .{ .exact_width = 8, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 1, .draft_prior = &.{case.accept}, .depth_rate = 0.02, .copy_drafts = false, .plain_guard = true, .hidden_rows = true, .window_costs = &costs }, 8, 7);
    defer cfg.deinit(a);
    var h: Target = .{ .case = case };
    defer h.target.deinit();
    var backend = h.target.backend();
    var vtable = backend.vtable.*;
    vtable.verify = Target.verify;
    vtable.draft = Target.draft;
    backend.vtable = &vtable;
    var e = Engine.init(a, &cfg, backend, .{ .ptr = &h, .vtable = &.{ .start = Target.start, .elapsed_ms = Target.elapsed } });
    defer e.deinit();
    var streams: [4]Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    for (&streams) |*s| {
        s.* = try Stream.init(a, .{ .id = "warm", .prompt = &.{ 1, 2, 3 }, .max_new = case.max_new, .eos = &.{}, .drafts = drafts, .sampling = if (sampled) .{ .seed = 23, .temperature = 0.7 } else null });
        made += 1;
        try e.addStream(s);
    }
    var out: Run = .{ .tokens = undefined };
    while (e.activeCount() > 0) {
        const four = e.activeCount() == 4;
        const due = drafts and four and try e.rule.shared_price.due(a, 4, sampled);
        try e.step();
        if (!drafts or !four or h.windows != 4) continue;
        const rows = h.target.last_rows;
        // a trial's streams hold no drafts, so the round after it verifies none
        const settled = h.fours > settled_from and (out.count == 0 or h.fours != out.trials[out.count - 1] + 1);
        if (due and rows == 4) {
            try t.expect(out.count < out.trials.len);
            out.trials[out.count] = h.fours;
            out.count += 1;
            const price = e.rule.sharedPrice(4, sampled);
            // the first trial is a warm-up: until the next one, the existing allocator stays in charge
            if (out.count == 1) try t.expect(price == null) else {
                const later: f64 = if (h.fours >= later_from) case.later else 0;
                try t.expectApproxEqAbs(plain_ms + later - forward_ms, price.?.plain, 1e-9);
            }
        } else if (settled) {
            out.rounds_after += 1;
            out.drafted_after += @intFromBool(rows > 4);
        }
    }
    for (&out.tokens, &streams) |*tokens, s| tokens.* = try a.dupe(u32, s.emitted());
    return out;
}

fn exact(case: Case, sampled: bool) !Run {
    var oracle = try run(case, sampled, false);
    defer oracle.deinit();
    var got = try run(case, sampled, true);
    errdefer got.deinit();
    for (oracle.tokens, got.tokens) |want, have| try t.expectEqualSlices(u32, want, have);
    return got;
}

test "shared prices: a slow first plain trial is dropped, so near-even drafts give way to plain rounds" {
    // an 8-row round costs 1.63x a warm plain one: one draft a stream pays above 0.63 acceptance, not at 0.6
    const case: Case = .{ .accept = 0.6, .row_ms = 0.63 * plain_ms / 4.0, .table_row_ms = 2.25, .max_new = 700 };
    for ([_]bool{ false, true }) |sampled| {
        var got = try exact(case, sampled);
        defer got.deinit();
        try t.expect(got.rounds_after > 200);
        try t.expectEqual(@as(usize, 0), got.drafted_after);
        try t.expectEqualSlices(u64, &.{ 64, 128 }, got.trials[0..got.count]);
    }
}

test "shared prices: profitable drafts keep winning, and each later trial replaces the plain price" {
    const case: Case = .{ .accept = 0.9, .row_ms = 0.5, .table_row_ms = 0.25, .later = 1, .max_new = 1400 };
    for ([_]bool{ false, true }) |sampled| {
        var got = try exact(case, sampled);
        defer got.deinit();
        // a missing price retries `every` rounds on, a measured one only once it is `stale`
        try t.expectEqualSlices(u64, &.{ 64, 128, 384, 640 }, got.trials[0..got.count]);
        try t.expectEqual(got.rounds_after, got.drafted_after);
    }
}
