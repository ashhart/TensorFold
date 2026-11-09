//! lookup.zig replays fixtures/lookup-golden.json round for round, then the same rounds through lookup.Proposer.
const std = @import("std");
const Value = std.json.Value;
const costs_mod = @import("costs.zig");
const lookup = @import("lookup.zig");

const gpa = std.testing.allocator;
const golden = @embedFile("fixtures/lookup-golden.json");

fn ints(a: std.mem.Allocator, v: Value) ![]u32 {
    const out = try a.alloc(u32, v.array.items.len);
    for (v.array.items, out) |x, *o| o.* = @intCast(x.integer);
    return out;
}

fn num(v: Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => unreachable,
    };
}

const Scenario = struct { costs: costs_mod.Costs, set: lookup.Settings, eos: []u32, prompt: []u32, rounds: []const Value };

fn scenario(a: std.mem.Allocator, s: Value) !Scenario {
    const o = s.object;
    const vs = o.get("costs").?.object.get("verify").?.array.items;
    const verify = try a.alloc(f64, vs.len);
    for (vs, verify) |x, *y| y.* = num(x);
    return .{
        .costs = .{ .verify = verify, .draft = num(o.get("costs").?.object.get("draft").?) },
        .set = .{ .plan = true, .min_match = o.get("min_match").?.integer, .max_rows = o.get("max_rows").?.integer },
        .eos = try ints(a, o.get("eos").?),
        .prompt = try ints(a, o.get("prompt").?),
        .rounds = o.get("rounds").?.array.items,
    };
}

fn kindOf(k: []const u8) lookup.Kind {
    return if (std.mem.eql(u8, k, "l")) .lookup else if (std.mem.eql(u8, k, "d")) .dspark else .serial;
}

test "the lookup plan replays Python's Planner round for round" {
    var parsed = try std.json.parseFromSlice(Value, gpa, golden, .{});
    defer parsed.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var rounds: usize = 0;
    var lookups: usize = 0;
    for (parsed.value.object.get("scenarios").?.array.items) |sv| {
        const sc = try scenario(a, sv);
        const pl = lookup.Planner.init(sc.set, &sc.costs, true);
        var r = try pl.start(gpa, sc.prompt);
        defer r.deinit();
        for (sc.rounds) |rv| {
            const o = rv.object;
            const want = try ints(a, o.get("drafts").?);
            const got = try pl.window(&r, o.get("left").?.integer, true, sc.eos);
            try std.testing.expectEqual(kindOf(o.get("kind").?.string), r.kind);
            try std.testing.expectEqualSlices(u32, want, got);
            try pl.observe(&r, @intCast(o.get("rows").?.integer), @intCast(o.get("kept").?.integer), try ints(a, o.get("emitted").?));
            rounds += 1;
            lookups += @intFromBool(r.kind == .lookup);
        }
    }
    try std.testing.expect(rounds > 3000 and lookups > 400);
}

test "the lanes proposer gives the Planner's drafts from the stream's context" {
    var parsed = try std.json.parseFromSlice(Value, gpa, golden, .{});
    defer parsed.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for (parsed.value.object.get("scenarios").?.array.items) |sv| {
        const sc = try scenario(a, sv);
        const pl = lookup.Planner.init(sc.set, &sc.costs, true);
        var x = try lookup.Proposer.init(&pl, gpa, sc.prompt, sc.eos);
        defer x.deinit();
        const p = x.proposer();
        var context: std.ArrayList(u32) = .empty;
        defer context.deinit(gpa);
        try context.appendSlice(gpa, sc.prompt);
        for (sc.rounds) |rv| {
            const o = rv.object;
            const want = try ints(a, o.get("drafts").?);
            const room = o.get("left").?.integer - 1;
            const got = try p.propose(context.items, room);
            try std.testing.expectEqualSlices(u32, want, got);
            // a second ask in the same round reads the decision back (no second count of a skipped band)
            try std.testing.expectEqualSlices(u32, want, try p.propose(context.items, room));
            p.round(@intCast(o.get("rows").?.integer), @intCast(o.get("kept").?.integer));
            try context.appendSlice(gpa, try ints(a, o.get("emitted").?));
        }
    }
}

test "copy drafter: continuation-ranked band matching prefers full continuation in the same band and avoids tail wrap" {
    var verify = [_]f64{ 42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 92, 97, 102, 107, 112, 117 };
    const c: costs_mod.Costs = .{ .verify = &verify, .draft = 5.6 };
    const span = [_]u32{ 10, 11, 12, 13, 14, 15, 16, 17, 18 };
    const cont = [_]u32{ 100, 101, 102, 103, 104, 105, 106, 107 };
    var prompt: std.ArrayList(u32) = .empty;
    defer prompt.deinit(gpa);
    // Earlier occurrence: 9-token match (band 1: 8..15) followed by 8 continuation tokens.
    try prompt.appendSlice(gpa, &span);
    try prompt.appendSlice(gpa, &cont);
    // Near-tail occurrence: 10-token match in band 1 (8..15) with only 1 continuation token (999) before the tail.
    try prompt.append(gpa, 9);
    try prompt.appendSlice(gpa, &span);
    try prompt.append(gpa, 999);
    try prompt.append(gpa, 9);
    try prompt.appendSlice(gpa, &span);

    // Baseline lookup (`copy_drafter = false`) picks the 10-token near-tail match and wraps into self-repetition.
    const pl_base = lookup.Planner.init(.{ .plan = true }, &c, true);
    var r_base = try pl_base.start(gpa, prompt.items);
    defer r_base.deinit();
    const ds_base = try pl_base.window(&r_base, 100, true, &.{});
    try std.testing.expect(ds_base.len > 0 and ds_base[0] == 999);

    // With `copy_drafter = true`, `matchRanked` stays in band 1 and picks the earlier 9-token match with 8 drafts.
    const pl_copy = lookup.Planner.init(.{ .plan = true, .copy_drafter = true }, &c, true);
    var r_copy = try pl_copy.start(gpa, prompt.items);
    defer r_copy.deinit();
    const ds_copy = try pl_copy.window(&r_copy, 100, true, &.{});
    try std.testing.expectEqual(lookup.Kind.lookup, r_copy.kind);
    try std.testing.expect(ds_copy.len > 0);
    try std.testing.expectEqualSlices(u32, cont[0..ds_copy.len], ds_copy);
}

test "copy drafter: adaptive doubling grows offer width 8 -> 15 on full acceptance and resets on partial rejection" {
    var verify: [16]f64 = undefined;
    for (&verify, 0..) |*v, i| v.* = 20.0 + 0.01 * @as(f64, @floatFromInt(i));
    const c: costs_mod.Costs = .{ .verify = &verify, .draft = 5.6 };
    const pl = lookup.Planner.init(.{ .plan = true, .copy_drafter = true }, &c, true);
    var prompt: std.ArrayList(u32) = .empty;
    defer prompt.deinit(gpa);
    for (0..32) |i| try prompt.append(gpa, @intCast(10 + i));
    for (0..20) |i| try prompt.append(gpa, @intCast(200 + i));
    for (0..32) |i| try prompt.append(gpa, @intCast(10 + i));

    var r = try pl.start(gpa, prompt.items);
    defer r.deinit();
    // Round 1: initial offer is capped at 8 drafts (`initial_offer = 8`).
    const d1 = try pl.window(&r, 100, true, &.{});
    try std.testing.expectEqual(@as(usize, 8), d1.len);
    for (d1, 0..) |tok, i| try std.testing.expectEqual(@as(u32, @intCast(200 + i)), tok);

    // Full acceptance of all 8 drafts (`rows = 9, kept = 8`) doubles the offer cap to 15 (`max_rows - 1`).
    try pl.observe(&r, 9, 8, &.{ 200, 201, 202, 203, 204, 205, 206, 207, 10 });
    try std.testing.expectEqual(@as(u5, 1), r.doublings);
    try r.index.extend(&.{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41 });
    const d2 = try pl.window(&r, 100, true, &.{});
    try std.testing.expectEqual(@as(usize, 15), d2.len);

    // Partial acceptance (`kept = 3 < drafts = 15`) resets `doublings` to 0.
    try pl.observe(&r, 16, 3, &.{ 200, 201, 202, 999 });
    try std.testing.expectEqual(@as(u5, 0), r.doublings);
}

test "copy drafter: survivalYield and altYield gate lookup drafts against neural drafter expected yield" {
    try std.testing.expectApproxEqAbs(@as(f64, 1.98), lookup.survivalYield(&.{ 0.9, 0.8, 0.5 }), 1e-6);
    var verify = [_]f64{ 42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 92, 97, 102, 107, 112, 117 };
    const c: costs_mod.Costs = .{ .verify = &verify, .draft = 5.6 };
    const pl = lookup.Planner.init(.{ .plan = true, .copy_drafter = true }, &c, true);
    const span = [_]u32{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
    const cont = [_]u32{ 30, 31, 32, 33, 34, 35, 36, 37 };
    var prompt: std.ArrayList(u32) = .empty;
    defer prompt.deinit(gpa);
    try prompt.appendSlice(gpa, &span);
    try prompt.appendSlice(gpa, &cont);
    try prompt.appendSlice(gpa, &span);

    var r = try pl.start(gpa, prompt.items);
    defer r.deinit();
    r.doublings = 2;
    // High neural survival yield (`14.0` expected drafts) beats a band-1 lookup window and resets `doublings`.
    r.dspark_yield = 14.0;
    const gated = try pl.window(&r, 100, true, &.{});
    try std.testing.expectEqual(@as(usize, 0), gated.len);
    try std.testing.expectEqual(lookup.Kind.dspark, r.kind);
    try std.testing.expectEqual(@as(u5, 0), r.doublings);

    // Lower neural survival yield (`0.8` expected drafts) lets the lookup copy window proceed.
    r.dspark_yield = lookup.survivalYield(&.{ 0.5, 0.6 });
    const open = try pl.window(&r, 100, true, &.{});
    try std.testing.expect(open.len > 0);
    try std.testing.expectEqual(lookup.Kind.lookup, r.kind);
}

test "copy drafter: lookup.Proposer through Lanes and Twin matches Twin.serial token for token" {
    const twin_mod = @import("twin.zig");
    const lanes_mod = @import("lanes.zig");
    const core_lanes = @import("lanes");
    const t = try twin_mod.Twin.init(gpa, .{}, 3);
    defer t.deinit();
    const d = try twin_mod.Drafter.init(gpa, t, 3);
    defer d.deinit();
    var c = try costs_mod.defaults(gpa, 64);
    defer c.deinit(gpa);
    for (c.verify, 0..) |*v, i| v.* = 20.0 + 0.01 * @as(f64, @floatFromInt(i));
    c.draft = 5.6;
    const x = try lanes_mod.Lanes.init(gpa, t.target(), d.pass(), c, .{ .shape = d.shape, .slots = 2 });
    defer x.deinit();

    const base_prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const serial_run = try t.serial(2, &base_prompt, 24, null, null);
    defer gpa.free(serial_run);

    // Construct a prompt containing the repeated pattern so lookup copy drafting activates on matching spans.
    var prompt: std.ArrayList(u32) = .empty;
    defer prompt.deinit(gpa);
    try prompt.appendSlice(gpa, &base_prompt);
    try prompt.appendSlice(gpa, serial_run[0..12]);
    try prompt.appendSlice(gpa, &base_prompt);

    for ([_]?core_lanes.Sampling{ null, .{ .temperature = 0.85, .top_p = 0.9, .top_k = 4, .seed = 42 } }) |sampling| {
        const want = try t.serial(2, prompt.items, 20, sampling, null);
        defer gpa.free(want);

        const pl = lookup.Planner.init(.{ .plan = true, .copy_drafter = true }, &x.policy.depth.costs, true);
        var prop = try lookup.Proposer.init(&pl, gpa, prompt.items, &.{});
        defer prop.deinit();

        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var cfg = try core_lanes.Config.init(gpa, try x.model(arena.allocator(), 64), 16, 15);
        defer cfg.deinit(gpa);
        var clock: core_lanes.fake.FixedClock = .{};
        var e = core_lanes.Engine.init(gpa, &cfg, x.backend(), clock.clock());
        defer e.deinit();
        x.attach(&e);
        var s = try core_lanes.Stream.init(gpa, .{ .id = "s0", .prompt = prompt.items, .max_new = 20, .sampling = sampling, .drafts = true, .proposer = prop.proposer() });
        defer s.deinit(gpa);
        try e.addStream(&s);
        while (e.activeCount() > 0) try e.step();

        try std.testing.expectEqualSlices(u32, want, s.emitted());
    }
}
