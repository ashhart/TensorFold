//! spec.zig replays fixtures/spec-golden.json bit for bit; on the CPU twin a speculated run == the run without it.
const std = @import("std");
const Value = std.json.Value;
const lanes = @import("lanes");
const twin = @import("twin.zig");
const ln = @import("lanes.zig");
const costs_mod = @import("costs.zig");
const spec = @import("spec.zig");
const iface = @import("iface.zig");
const dspark = @import("dspark.zig");

const gpa = std.testing.allocator;
const golden = @embedFile("fixtures/spec-golden.json");

test "the knobs parse as Python's" {
    try std.testing.expectEqual(spec.Settings{}, try spec.Settings.parse(null, null));
    try std.testing.expectEqual(spec.Settings{ .on = true }, try spec.Settings.parse(" On", ""));
    try std.testing.expectEqual(spec.Settings{ .on = true, .nucleus = true }, try spec.Settings.parse("true", "1"));
    try std.testing.expectEqual(spec.Settings{}, try spec.Settings.parse("0", "off"));
    try std.testing.expectError(error.BadSpecKnob, spec.Settings.parse("2", null));
    try std.testing.expect(spec.isNucleus(.{ .seed = 1, .temperature = 0.7, .top_k = 0, .top_p = 0.9 }));
    try std.testing.expect(!spec.isNucleus(.{ .seed = 1, .temperature = 0.7, .top_k = 20, .top_p = 0.9 }));
    try std.testing.expect(!spec.isNucleus(.{ .seed = 1, .temperature = 0.7, .top_k = 0, .top_p = 1.0 }));
    try std.testing.expect(!spec.isNucleus(null));
}

fn int(v: Value) u64 {
    return @intCast(v.integer);
}

/// The golden's keyed noise: [] greedy, [seed] sampled (compared as params).
fn samplingOf(v: Value) ?lanes.Sampling {
    if (v.array.items.len == 0) return null;
    return .{ .seed = int(v.array.items[0]), .temperature = 0.7, .top_k = 20, .top_p = 0.95 };
}

fn expectState(s: *const spec.Spec, want: Value) !void {
    const o = want.object;
    for (o.get("rates").?.array.items, s.rate) |w, g| try std.testing.expectEqual(switch (w) {
        .float => |f| f,
        .integer => |i| @as(f64, @floatFromInt(i)),
        else => unreachable,
    }, g);
    for (o.get("skipped").?.array.items, s.skipped) |w, g| try std.testing.expectEqual(int(w), g);
    try std.testing.expectEqual(int(o.get("launched").?), s.stats.launched);
    try std.testing.expectEqual(int(o.get("hit").?), s.stats.hit);
    try std.testing.expectEqual(int(o.get("differs").?), s.stats.commit_differs);
    const miss = o.get("miss").?.object;
    for (std.enums.values(spec.Why)) |y| {
        const w: u64 = if (miss.get(@tagName(y))) |v| int(v) else 0;
        try std.testing.expectEqual(w, s.stats.miss.get(y));
    }
}

test "the decisions replay spec.py's (launch, ingested, take, forget, clear, rates, probes)" {
    var parsed = try std.json.parseFromSlice(Value, gpa, golden, .{});
    defer parsed.deinit();
    try std.testing.expectEqual(spec.decay, parsed.value.object.get("decay").?.float);
    try std.testing.expectEqual(spec.min_rate, parsed.value.object.get("min_rate").?.float);
    try std.testing.expectEqual(@as(u64, spec.probe), int(parsed.value.object.get("probe").?));
    var s = try spec.Spec.init(gpa, .{ .on = true }, 4, 5, 1);
    defer s.deinit();
    var drafts: [4][5]u32 = undefined;
    var conf: [4][5]f32 = undefined;
    for (parsed.value.object.get("ops").?.array.items) |op| {
        const o = op.object;
        const kind = o.get("op").?.string;
        if (std.mem.eql(u8, kind, "forget")) {
            s.forget(@intCast(int(o.get("slot").?)));
        } else if (std.mem.eql(u8, kind, "clear")) {
            s.clear();
        } else if (std.mem.eql(u8, kind, "launch")) {
            var cands: [4]spec.Candidate = undefined;
            const items = o.get("cands").?.array.items;
            for (items, cands[0..items.len]) |c, *x| {
                const a = c.array.items;
                x.* = .{ .slot = @intCast(int(a[0])), .start = int(a[1]), .accepted = @intCast(int(a[2])), .bonus = @intCast(int(a[3])), .sampling = samplingOf(a[4]) };
            }
            const asks = s.launch(cands[0..items.len]);
            try std.testing.expectEqual(o.get("got").?.bool, asks.len > 0);
        } else if (std.mem.eql(u8, kind, "ingested")) {
            for (o.get("commits").?.array.items) |c| {
                const a = c.array.items;
                var path: [16]u32 = undefined;
                const n = int(a[2]) + 1;
                for (path[0..n], 0..) |*p, i| p.* = @intCast(i);
                try std.testing.expectEqual(a[3].bool, s.ingested(@intCast(int(a[0])), int(a[1]), path[0..n]));
            }
        } else {
            const items = o.get("asks").?.array.items;
            var asks: [4]iface.Ask = undefined;
            var out: [4]iface.Proposal = undefined;
            for (items, asks[0..items.len], out[0..items.len], 0..) |c, *x, *p, i| {
                const a = c.array.items;
                const start = int(a[2]);
                x.* = .{ .slot = @intCast(int(a[0])), .anchor = @intCast(int(a[1])), .start = start, .params = dspark.Params.of(samplingOf(a[3]), start) };
                p.* = .{ .drafts = &drafts[i], .conf = &conf[i] };
            }
            try std.testing.expectEqual(o.get("got").?.bool, s.take(asks[0..items.len], out[0..items.len]));
        }
        try expectState(&s, o.get("state").?);
    }
}

const Case = struct { prompt: []const u32, max_new: u32 = 70, sampling: ?lanes.Sampling = null };

const Run = struct {
    out: [][]u32,
    drafted: u64,
    accepted: u64,
    kept: [16]u64,
    passes: u64,
    ingests: u64,
    stats: spec.Stats,

    fn deinit(r: Run) void {
        for (r.out) |o| gpa.free(o);
        gpa.free(r.out);
    }
};

const Mode = enum { off, sync, device };

fn run(cases: []const Case, mode: Mode, settings: spec.Settings, siblings: u32) !Run {
    const dims: twin.Dims = .{};
    const slots: u32 = 4;
    const t = try twin.Twin.init(gpa, dims, slots + 1);
    defer t.deinit();
    const d = try twin.Drafter.init(gpa, t, slots + 1);
    defer d.deinit();
    var c = try @import("lanes_test.zig").servedCosts(gpa);
    defer c.deinit(gpa);
    const pass = if (mode == .device) d.passAsync() else d.pass();
    var set = settings;
    set.on = mode != .off;
    const x = try ln.Lanes.init(gpa, t.target(), pass, c, .{ .shape = d.shape, .slots = slots, .siblings = siblings, .spec = set });
    defer x.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cfg = try lanes.Config.init(gpa, try x.model(arena.allocator(), 64), 16, 15);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpa, &cfg, x.backend(), clock.clock());
    defer e.deinit();
    x.attach(&e);
    const streams = try gpa.alloc(lanes.Stream, cases.len);
    defer gpa.free(streams);
    for (cases, streams) |cs, *s| s.* = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = cs.prompt, .max_new = cs.max_new, .sampling = cs.sampling, .drafts = true });
    defer for (streams) |*s| s.deinit(gpa);
    for (streams) |*s| try e.addStream(s);
    while (e.activeCount() > 0) try e.step();
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return .{ .out = out, .drafted = e.drafted, .accepted = e.accepted, .kept = x.policy.depth.stats.kept, .passes = d.passes, .ingests = d.ingests, .stats = if (x.spec) |sp| sp.stats else .{} };
}

fn expectSame(cases: []const Case, settings: spec.Settings, siblings: u32) ![3]spec.Stats {
    const off = try run(cases, .off, settings, siblings);
    defer off.deinit();
    for (cases, off.out) |cs, got| { // the reference itself is the serial decode
        const t = try twin.Twin.init(gpa, .{}, 1);
        defer t.deinit();
        const want = try t.serial(0, cs.prompt, cs.max_new, cs.sampling, null);
        defer gpa.free(want);
        try std.testing.expectEqualSlices(u32, want, got);
    }
    var stats: [3]spec.Stats = undefined;
    for ([_]Mode{ .sync, .device }, 1..) |m, i| {
        const r = try run(cases, m, settings, siblings);
        defer r.deinit();
        for (off.out, r.out) |a, b| try std.testing.expectEqualSlices(u32, a, b);
        // the same drafts: the same windows, acceptance and kept counts as the run without speculation
        try std.testing.expectEqual(off.drafted, r.drafted);
        try std.testing.expectEqual(off.accepted, r.accepted);
        try std.testing.expectEqualSlices(u64, &off.kept, &r.kept);
        // every used speculation replaced a pass, every launch ran one; skipped commits saved their ingests
        try std.testing.expectEqual(off.passes + r.stats.launched - r.stats.hit, r.passes);
        try std.testing.expectEqual(off.ingests + r.stats.launched_slots, r.ingests + r.stats.ingests_skipped);
        stats[i] = r.stats;
    }
    stats[0] = .{};
    return stats;
}

const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5 };
const p3 = [_]u32{ 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30 };
const hot: lanes.Sampling = .{ .seed = 11, .temperature = 0.8, .top_k = 0, .top_p = 0.9 };
const warm: lanes.Sampling = .{ .seed = 5, .temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0.02 };

test "one slot: speculated == unspeculated == serial, greedy and keyed, most speculations used" {
    for ([_]?lanes.Sampling{ null, warm }) |s| {
        const st = try expectSame(&.{.{ .prompt = &p1, .sampling = s, .max_new = 90 }}, .{}, 0);
        for (st[1..]) |x| try std.testing.expect(x.launched > 0 and x.hit * 2 > x.launched and x.ingests_skipped > 0);
    }
}

test "shared rounds over several slots: one pass serves the speculated slots" {
    const cases = [_]Case{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = warm, .max_new = 50 }, .{ .prompt = &p3 } };
    const st = try expectSame(&cases, .{}, 0);
    for (st[1..]) |x| try std.testing.expect(x.launched > 0 and x.hit > 0);
}

test "nucleus rows speculate only with TF_DSV41_SPEC_NUCLEUS" {
    const cases = [_]Case{.{ .prompt = &p2, .sampling = hot, .max_new = 60 }};
    const off = try expectSame(&cases, .{}, 0);
    for (off[1..]) |x| try std.testing.expectEqual(@as(u64, 0), x.launched);
    const on = try expectSame(&cases, .{ .nucleus = true }, 0);
    for (on[1..]) |x| try std.testing.expect(x.launched > 0 and x.hit > 0);
}

test "tree rounds are not speculated, the chain rounds around them are" {
    _ = try expectSame(&.{.{ .prompt = &p2, .max_new = 120 }}, .{}, 2);
}
