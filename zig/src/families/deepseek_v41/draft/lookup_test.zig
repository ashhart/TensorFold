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
