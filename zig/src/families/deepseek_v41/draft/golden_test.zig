//! Reference decisions replayed (fixtures/golden.json): depth, joint, trees, calibration, rates, bit for bit.
const std = @import("std");
const Value = std.json.Value;
const costs_mod = @import("costs.zig");
const depth = @import("depth.zig");
const joint = @import("joint.zig");
const tree = @import("tree.zig");

const gpa = std.testing.allocator;
const golden = @embedFile("fixtures/golden.json");

fn num(v: Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => unreachable,
    };
}

fn int(v: Value) usize {
    return @intCast(v.integer);
}

fn floats(a: std.mem.Allocator, v: Value) ![]f64 {
    const out = try a.alloc(f64, v.array.items.len);
    for (out, v.array.items) |*o, x| o.* = num(x);
    return out;
}

fn opt(v: Value) ?u32 {
    return if (v == .null) null else @intCast(v.integer);
}

fn expectFloats(want: Value, got: []const f64) !void {
    for (want.array.items, got) |w, g| try std.testing.expectEqual(num(w), g);
}

fn costs(a: std.mem.Allocator, root: Value) !costs_mod.Costs {
    const c = root.object.get("costs").?;
    return .{ .verify = try floats(a, c.object.get("verify").?), .draft = num(c.object.get("draft").?), .slot = num(c.object.get("slot").?) };
}

test "the cost table equals calib.py's from the same timings" {
    var parsed = try std.json.parseFromSlice(Value, gpa, golden, .{});
    defer parsed.deinit();
    var mine = try @import("lanes_test.zig").servedCosts(gpa);
    defer mine.deinit(gpa);
    const want = try costs(parsed.arena.allocator(), parsed.value);
    try std.testing.expectEqualSlices(f64, want.verify, mine.verify);
    try std.testing.expectEqual(want.draft, mine.draft);
}

test "depth traces replay the Python decisions and state" {
    var parsed = try std.json.parseFromSlice(Value, gpa, golden, .{});
    defer parsed.deinit();
    const a = parsed.arena.allocator();
    const c = try costs(a, parsed.value);
    for (parsed.value.object.get("traces").?.array.items) |trace| {
        var d = try depth.Depth.init(gpa, c, 5, .{ .joint = @intCast(trace.object.get("joint").?.integer) });
        defer d.deinit();
        for (trace.object.get("ops").?.array.items) |op| {
            const o = op.object;
            const kind = o.get("op").?.string;
            if (std.mem.eql(u8, kind, "joint")) {
                const items = o.get("asks").?.array.items;
                const asks = try a.alloc(depth.Depth.Ask, items.len);
                for (asks, items) |*x, it| x.* = .{ .slot = @intCast(it.object.get("slot").?.integer), .conf = try floats(a, it.object.get("conf").?), .most = int(it.object.get("most").?), .first = opt(it.object.get("first").?) };
                var others: u64 = 0;
                for (o.get("others").?.array.items) |r| others += int(r);
                const ks = try a.alloc(usize, asks.len);
                try d.chooseJoint(asks, others, o.get("others").?.array.items.len, null, ks);
                for (o.get("ks").?.array.items, ks) |w, g| try std.testing.expectEqual(int(w), g);
            } else if (std.mem.eql(u8, kind, "tree")) {
                const sibs = try floats(a, o.get("sibs").?);
                const got = try d.chooseTree(@intCast(o.get("slot").?.integer), try floats(a, o.get("conf").?), int(o.get("most").?), sibs, true, opt(o.get("first").?), .{ .ms = 0.4, .rows = true }, 4);
                try std.testing.expectEqual(int(o.get("k").?), got.k);
                for (o.get("lens").?.array.items, got.lens[0..sibs.len]) |w, g| try std.testing.expectEqual(int(w), g);
            } else if (std.mem.eql(u8, kind, "record")) {
                const slot: u32 = @intCast(o.get("slot").?.integer);
                const tokens = o.get("tokens").?;
                try d.record(slot, int(o.get("rows").?), int(o.get("keep").?), if (tokens == .null) null else int(tokens));
                d.bonus(slot, @intCast(o.get("bonus").?.integer));
            } else if (std.mem.eql(u8, kind, "reset")) {
                d.reset(@intCast(o.get("slot").?.integer));
            } else {
                try expectFloats(o.get("kept").?, d.cal.kept);
                try expectFloats(o.get("prob").?, d.cal.prob);
                for (o.get("rates").?.array.items, 0..) |w, s| try std.testing.expectEqual(num(w), try d.rate(@intCast(s)));
                for (o.get("joint").?.array.items, 1..) |w, n| try std.testing.expectEqual(num(w), d.jointRate(n));
                try std.testing.expectEqual(@as(u64, @intCast(o.get("zero").?.integer)), d.stats.zero);
                for (o.get("reached").?.array.items, o.get("statkept").?.array.items, 0..) |r, k, j| {
                    try std.testing.expectEqual(@as(u64, @intCast(r.integer)), d.stats.reached[j]);
                    try std.testing.expectEqual(@as(u64, @intCast(k.integer)), d.stats.kept[j]);
                }
            }
        }
    }
}

test "tree plans, joint allocations and fairness weights equal Python's" {
    var parsed = try std.json.parseFromSlice(Value, gpa, golden, .{});
    defer parsed.deinit();
    const a = parsed.arena.allocator();
    const c = try costs(a, parsed.value);
    for (parsed.value.object.get("plans").?.array.items) |p| {
        const o = p.object;
        const got = tree.plan(try floats(a, o.get("qm").?), try floats(a, o.get("sibs").?), try floats(a, o.get("cont").?), c.verify, num(o.get("rate").?), int(o.get("least").?), .{ .ms = num(o.get("dup").?), .rows = true }, int(o.get("rows").?), 16);
        try std.testing.expectEqual(int(o.get("k").?), got.k);
        for (o.get("lens").?.array.items, 0..) |w, j| try std.testing.expectEqual(int(w), got.lens[j]);
        try std.testing.expectEqual(num(o.get("ms").?), got.ms);
        try std.testing.expectEqual(num(o.get("e").?), got.expected);
    }
    for (parsed.value.object.get("allocs").?.array.items) |p| {
        const o = p.object;
        const qs = o.get("qs").?.array.items;
        const slots = try a.alloc(joint.Slot, qs.len);
        const w = o.get("w").?;
        for (slots, qs, 0..) |*s, q, i| s.* = .{ .qs = try floats(a, q), .least = int(o.get("lo").?.array.items[i]), .top = int(o.get("top").?.array.items[i]), .weight = if (w == .null) 1.0 else num(w.array.items[i]) };
        const ks = try a.alloc(usize, qs.len);
        const best = try joint.allocate(gpa, slots, int(o.get("shared").?), c, num(o.get("rate").?), int(o.get("max").?), ks);
        for (o.get("ks").?.array.items, ks) |x, k| try std.testing.expectEqual(int(x), k);
        try std.testing.expectEqual(num(o.get("best").?), best);
    }
    for (parsed.value.object.get("weights").?.array.items) |p| {
        const o = p.object;
        const items = o.get("tpr").?.array.items;
        const tpr = try a.alloc(?f64, items.len);
        for (tpr, items) |*t, x| t.* = if (x == .null) null else num(x);
        const out = try a.alloc(f64, items.len);
        joint.weights(tpr, num(o.get("alpha").?), out);
        try expectFloats(o.get("w").?, out);
    }
}
