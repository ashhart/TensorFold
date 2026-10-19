//! calib_measure.zig against fixtures/calib-measure-golden.json: same runs, same gathered ints, same Costs bits.
const std = @import("std");
const Value = std.json.Value;
const cm = @import("calib_measure.zig");
const costs_mod = @import("costs.zig");

const gpa = std.testing.allocator;
const golden = @embedFile("fixtures/calib-measure-golden.json");

fn num(v: Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => unreachable,
    };
}

fn floats(a: std.mem.Allocator, v: Value) ![]f64 {
    const out = try a.alloc(f64, v.array.items.len);
    for (out, v.array.items) |*o, x| o.* = num(x);
    return out;
}

const Env = struct {
    var obj: ?std.json.ObjectMap = null;
    fn get(name: [*:0]const u8) ?[]const u8 {
        const o = obj orelse return null;
        return if (o.get(std.mem.span(name))) |v| v.string else null;
    }
};

fn methodOf(env: Value) !cm.Method {
    Env.obj = env.object;
    defer Env.obj = null;
    return cm.method(&Env.get);
}

test "the measurement equals Python's calib.measure: runs, gathered ints and the table" {
    var parsed = try std.json.parseFromSlice(Value, gpa, golden, .{});
    defer parsed.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = parsed.value.object.get("cases").?.array.items;
    try std.testing.expect(cases.len >= 5);
    for (cases) |cv| {
        const c = cv.object;
        const m = try methodOf(c.get("env").?);
        const slots: u32 = @intCast(c.get("slots").?.integer);
        var p = try cm.Plan.init(gpa, slots, @intCast(c.get("cap").?.integer), cm.max_rows, m, c.get("draft").?.bool);
        defer p.deinit(gpa);
        try std.testing.expectEqual(@as(u32, @intCast(c.get("n").?.integer)), p.n);
        const wide = c.get("wide").?.array.items;
        try std.testing.expectEqual(wide.len, p.wideBucketsOf().len);
        for (wide, p.wideBucketsOf()) |w, b| try std.testing.expectEqual(@as(u32, @intCast(w.integer)), b);
        // the same runs in the same order: each window's (slot, rows) segments, or the DSpark pass
        const calls = c.get("calls").?.array.items;
        try std.testing.expectEqual(calls.len, p.runs.items.len);
        for (calls, p.runs.items) |call, run| {
            if (call == .string) {
                try std.testing.expectEqualStrings("draft", call.string);
                try std.testing.expect(run.what == .draft and run.nsegs == 0);
                continue;
            }
            const segs = call.array.items;
            try std.testing.expectEqual(segs.len, run.nsegs);
            for (segs, run.segments()) |s, r| {
                try std.testing.expectEqual(@as(u32, @intCast(s.array.items[0].integer)), r.slot);
                try std.testing.expectEqual(@as(u32, @intCast(s.array.items[1].integer)), r.rows);
            }
        }
        // each rank's kept values as the golden's gathered ints, then the table from both
        var ranks: [2][]f64 = undefined;
        for (0..2) |rk| {
            const samples = try floats(a, c.get("samples").?.array.items[rk]);
            ranks[rk] = try a.alloc(f64, p.entries());
            try p.values(samples, ranks[rk]);
            const ints = c.get("ints").?.array.items[rk].array.items;
            try std.testing.expectEqual(ints.len, ranks[rk].len);
            for (ints, ranks[rk]) |want, got| try std.testing.expectEqual(want.integer, costs_mod.roundEven(got * 1000.0));
        }
        var got = try p.build(gpa, &.{ ranks[0], ranks[1] });
        defer got.deinit(gpa);
        const want = c.get("costs").?.object;
        const verify = want.get("verify").?.array.items;
        try std.testing.expectEqual(verify.len, got.verify.len);
        for (verify, got.verify) |x, y| try std.testing.expectEqual(num(x), y);
        try std.testing.expectEqual(num(want.get("draft").?), got.draft);
        try std.testing.expectEqual(num(want.get("slot").?), got.slot);
        // and as the ranks share a cached table (Costs.encode / decode): microseconds, half to even
        const ints = try got.encode(gpa);
        defer gpa.free(ints);
        var back = try costs_mod.Costs.decode(gpa, ints);
        defer back.deinit(gpa);
        for (back.verify, got.verify) |x, y| try std.testing.expectEqual(@as(f64, @floatFromInt(@max(1, costs_mod.roundEven(y * 1000.0)))) / 1000.0, x);
    }
}

test "method, statistic, table_of, extend, split, wide_buckets and fallback_ids equal Python's" {
    var parsed = try std.json.parseFromSlice(Value, gpa, golden, .{});
    defer parsed.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const pure = parsed.value.object.get("pure").?.object;
    for (pure.get("statistic").?.array.items) |x| {
        const xs = try floats(a, x.object.get("xs").?);
        try std.testing.expectEqual(num(x.object.get("median").?), try cm.statistic(xs, .median));
        try std.testing.expectEqual(num(x.object.get("min").?), try cm.statistic(xs, .min));
    }
    for (pure.get("table").?.array.items) |x| {
        const times = try floats(a, x.object.get("times").?);
        for ([_]cm.Shape{ .raw, .fit }) |sh| {
            const want = try floats(a, x.object.get(@tagName(sh)).?);
            const got = try cm.tableOf(gpa, times, sh);
            defer gpa.free(got);
            try std.testing.expectEqualSlices(f64, want, got);
        }
    }
    for (pure.get("extend").?.array.items) |x| {
        const table = try floats(a, x.object.get("table").?);
        const pv = x.object.get("points").?.array.items;
        const pts = try a.alloc(costs_mod.Point, pv.len);
        for (pts, pv) |*pt, v| pt.* = .{ .rows = @intCast(v.array.items[0].integer), .ms = num(v.array.items[1]) };
        const got = try costs_mod.extend(gpa, table, pts);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(f64, try floats(a, x.object.get("out").?), got);
    }
    for (pure.get("split").?.array.items) |x| {
        var out: [8]u32 = undefined;
        const n: usize = @intCast(x.object.get("n").?.integer);
        cm.split(@intCast(x.object.get("total").?.integer), out[0..n]);
        for (x.object.get("out").?.array.items, out[0..n]) |w, g| try std.testing.expectEqual(@as(u32, @intCast(w.integer)), g);
    }
    for (pure.get("wide").?.array.items) |x| {
        var buf: [4]u32 = undefined;
        const got = cm.wideBuckets(@intCast(x.object.get("slots").?.integer), @intCast(x.object.get("cap").?.integer), cm.max_rows, &buf);
        const want = x.object.get("out").?.array.items;
        try std.testing.expectEqual(want.len, got.len);
        for (want, got) |w, g| try std.testing.expectEqual(@as(u32, @intCast(w.integer)), g);
    }
    try std.testing.expectEqualStrings(pure.get("text").?.string, cm.text); // the calibration text
    var methods: usize = 0;
    for (pure.get("method").?.array.items) |x| {
        const got = methodOf(x.object.get("env").?);
        const want = x.object.get("out").?;
        if (want == .null) {
            if (got) |_| return error.TestExpectedError else |_| {}
            continue;
        }
        const m = try got;
        const o = want.object;
        try std.testing.expectEqual(@as(u32, @intCast(o.get("reps").?.integer)), m.reps);
        try std.testing.expectEqual(@as(u32, @intCast(o.get("warm").?.integer)), m.warm);
        try std.testing.expectEqual(@as(u32, @intCast(o.get("recheck").?.integer)), m.recheck);
        try std.testing.expectEqual(@as(u32, @intCast(o.get("cycle").?.integer)), m.cycle);
        try std.testing.expectEqual(@as(u32, @intCast(o.get("deep").?.integer)), m.deep);
        try std.testing.expectEqualStrings(o.get("stat").?.string, @tagName(m.stat));
        try std.testing.expectEqualStrings(o.get("shape").?.string, @tagName(m.shape));
        methods += 1;
    }
    try std.testing.expect(methods >= 4);
    for (pure.get("fallback").?.array.items) |x| {
        var buf: [64]u32 = undefined;
        const got = cm.fallbackIds(@intCast(x.object.get("vocab").?.integer), &buf);
        for (x.object.get("out").?.array.items, got) |w, g| try std.testing.expectEqual(@as(u32, @intCast(w.integer)), g);
    }
}

test "the calibration prompt: ids past the vocabulary dropped, at most 256, fallback under 16; slot cuts" {
    var ids: [300]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(if (i % 10 == 3) 200_000 else i);
    var fb: [64]u32 = undefined;
    const got = cm.promptIds(&ids, 129280, &fb);
    try std.testing.expectEqual(@as(usize, 256), got.len);
    try std.testing.expectEqual(@as(u32, 4), got[3]);
    var few = [_]u32{ 1, 2, 3 };
    try std.testing.expectEqual(@as(usize, 64), cm.promptIds(&few, 129280, &fb).len);
    try std.testing.expectEqual(@as(usize, 120), cm.cutOf(120, 0));
    try std.testing.expectEqual(@as(usize, 113), cm.cutOf(120, 1));
    try std.testing.expectEqual(@as(usize, 20), cm.cutOf(20, 2)); // 20 > 14 + 8 fails: the whole prompt
    try std.testing.expect(std.mem.endsWith(u8, cm.text, "and the") and std.mem.startsWith(u8, cm.text, "The committee"));
}
