//! calib_env.zig against fixtures/calib-golden.json: same entry picked, same shared table bits, same depth and tree.
const std = @import("std");
const Value = std.json.Value;
const calib_env = @import("calib_env.zig");
const depth = @import("depth.zig");
const costs_mod = @import("costs.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;
const golden = @embedFile("fixtures/calib-golden.json");

fn num(v: Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => unreachable,
    };
}

test "a served cached calibration: the entry, the shared table and every depth choice equal Python's" {
    var parsed = try std.json.parseFromSlice(Value, gpa, golden, .{});
    defer parsed.deinit();
    const g = parsed.value.object;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    for (g.get("files").?.array.items) |f| {
        const content = try std.json.Stringify.valueAlloc(a, f.object.get("content").?, .{});
        try tmp.dir.writeFile(io, .{ .sub_path = f.object.get("name").?.string, .data = content });
    }
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const w = g.get("want").?.object;
    const want: calib_env.Shape = .{ .slots = @intCast(w.get("slots").?.integer), .world = @intCast(w.get("world").?.integer), .dspark = w.get("dspark").?.bool, .context = @intCast(w.get("context").?.integer) };
    var got = try calib_env.load(gpa, io, .{ .dir = root }, want, 16);
    defer got.deinit(gpa);
    try std.testing.expectEqual(calib_env.Source.cached, got.source);
    try std.testing.expectEqualStrings(g.get("pick").?.string, std.fs.path.basename(got.path.?));
    const c = g.get("costs").?.object;
    const verify = c.get("verify").?.array.items;
    try std.testing.expectEqual(verify.len, got.costs.verify.len);
    for (verify, got.costs.verify) |x, y| try std.testing.expectEqual(num(x), y);
    try std.testing.expectEqual(num(c.get("draft").?), got.costs.draft);
    try std.testing.expectEqual(num(c.get("slot").?), got.costs.slot);
    // every ask on a fresh depth (block=5, mode="cost", cap=5, skip=True, joint=1)
    var trees: usize = 0;
    for (g.get("asks").?.array.items) |ask| {
        const o = ask.object;
        var d = try depth.Depth.init(gpa, got.costs, 5, .{});
        defer d.deinit();
        var conf: [5]f64 = undefined;
        for (&conf, o.get("conf").?.array.items) |*q, x| q.* = num(x);
        const most: usize = @intCast(o.get("most").?.integer);
        const first: ?u32 = if (o.get("first").? == .null) null else @intCast(o.get("first").?.integer);
        if (o.get("sibs")) |sv| {
            var sibs: [3]f64 = undefined;
            for (sv.array.items, 0..) |x, i| sibs[i] = num(x);
            const t = try d.chooseTree(0, &conf, most, sibs[0..sv.array.items.len], true, first, .{ .ms = 0.4, .rows = true }, 4);
            try std.testing.expectEqual(@as(usize, @intCast(o.get("k").?.integer)), t.k);
            for (o.get("lens").?.array.items, 0..) |l, i| try std.testing.expectEqual(@as(usize, @intCast(l.integer)), t.lens[i]);
            trees += 1;
        } else try std.testing.expectEqual(@as(usize, @intCast(o.get("k").?.integer)), try d.choose(0, &conf, most, true, first));
    }
    try std.testing.expect(trees > 0);
}

test "no entry for this engine: the defaults; more slots fit; a named file wins; the modes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [256]u8 = undefined;
    const root = try std.fmt.bufPrint(&buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const want: calib_env.Shape = .{ .slots = 1, .world = 2, .dspark = true };
    var none = try calib_env.load(gpa, io, .{ .dir = root }, want, 16);
    defer none.deinit(gpa);
    try std.testing.expectEqual(calib_env.Source.default, none.source);
    try std.testing.expectEqual(@as(f64, 42.0), none.costs.verify[0]); // depth.default_costs: 37 + 5 a row
    try tmp.dir.writeFile(io, .{ .sub_path = "calib-x.json", .data = "{\"verify\": [20.0004999, 25.0005], \"draft\": 3.0, \"meta\": {\"shape\": {\"slots\": 1, \"world\": 1, \"dspark\": true}}}" });
    var wrong = try calib_env.load(gpa, io, .{ .dir = root }, want, 16); // world 1: not ours
    defer wrong.deinit(gpa);
    try std.testing.expectEqual(calib_env.Source.default, wrong.source);
    var p: [300]u8 = undefined;
    var named = try calib_env.load(gpa, io, .{ .dir = root, .file = try std.fmt.bufPrint(&p, "{s}/calib-x.json", .{root}) }, want, 16);
    defer named.deinit(gpa);
    try std.testing.expectEqualSlices(f64, &.{ 20.0, 25.0 }, named.costs.verify); // microseconds, half to even
    var dflt = try calib_env.load(gpa, io, .{ .mode = "Default", .dir = root }, want, 16);
    defer dflt.deinit(gpa);
    try std.testing.expectEqual(calib_env.Source.default, dflt.source);
    // TF_DSV41_CALIB=real with a 4-slot table: a 1-slot engine reads its newest
    try tmp.dir.writeFile(io, .{ .sub_path = "calib-p.json", .data = "{\"verify\": [21.0, 26.0], \"draft\": 3.6, \"meta\": {\"shape\": {\"slots\": 4, \"world\": 2, \"dspark\": true}, \"time\": \"2026-10-07T12:00:00\"}}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "calib-q.json", .data = "{\"verify\": [22.0, 27.0], \"draft\": 3.6, \"meta\": {\"shape\": {\"slots\": 4, \"world\": 2, \"dspark\": true}, \"time\": \"2026-10-06T12:00:00\"}}" });
    var served = try calib_env.load(gpa, io, .{ .mode = "real", .dir = root }, want, 16);
    defer served.deinit(gpa);
    try std.testing.expectEqualStrings("calib-p.json", std.fs.path.basename(served.path.?));
    try std.testing.expectError(error.CalibMode, calib_env.load(gpa, io, .{ .mode = "measured" }, want, 16));
}

test "code acceptance is default off and rejects invalid flags" {
    Env.pairs = &.{};
    try std.testing.expect(!(try calib_env.depthSettings(&Env.get)).code_accept);
    Env.pairs = &.{.{ "TF_DSV41_CODE_ACCEPT", "on" }};
    try std.testing.expect((try calib_env.depthSettings(&Env.get)).code_accept);
    Env.pairs = &.{.{ "TF_DSV41_CODE_ACCEPT", "maybe" }};
    try std.testing.expectError(error.CodeAccept, calib_env.depthSettings(&Env.get));
    Env.pairs = &.{};
}

const Env = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: [*:0]const u8) ?[]const u8 {
        for (pairs) |kv| if (std.mem.eql(u8, kv[0], std.mem.span(name))) return kv[1];
        return null;
    }
};

test "the depth's knobs read as Python's" {
    Env.pairs = &.{};
    const d = try calib_env.depthSettings(&Env.get);
    try std.testing.expect(d.mode == .cost and d.cap == 5 and d.skip and d.joint == 1);
    Env.pairs = &.{ .{ "TF_DSV41_DEPTH", "static" }, .{ "TF_DSV41_DRAFT_SKIP", "off" }, .{ "TF_DSV41_DEPTH_JOINT", "2" }, .{ "TF_DSV41_DEPTH_FAIR", "1.5" }, .{ "TF_DSV41_GRAPH_ROWS_MAX", "32" } };
    const s = try calib_env.depthSettings(&Env.get);
    try std.testing.expect(s.mode == .static and s.cap == 3 and !s.skip and s.joint == 2 and s.fair == 1.5 and s.max_rows == 32);
    Env.pairs = &.{.{ "TF_DSV41_DRAFT_DEPTH", "16" }};
    try std.testing.expectError(error.DraftDepth, calib_env.depthSettings(&Env.get));
    Env.pairs = &.{.{ "TF_DSV41_DEPTH_JOINT", "3" }};
    try std.testing.expectError(error.DepthJoint, calib_env.depthSettings(&Env.get));
}

test "TF_DSV41_CALIB=measure / zig: the engine measures; a stored Zig table is read back by zig only" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [256]u8 = undefined;
    const root = try std.fmt.bufPrint(&buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const want: calib_env.Shape = .{ .slots = 4, .world = 2, .dspark = true, .context = 1048576 };
    var m = try calib_env.load(gpa, io, .{ .mode = "Measure", .dir = root }, want, 64);
    defer m.deinit(gpa);
    try std.testing.expectEqual(calib_env.Source.measure, m.source);
    try std.testing.expectEqual(@as(usize, 64), m.costs.verify.len); // the defaults until measured
    var none = try calib_env.load(gpa, io, .{ .mode = "zig", .dir = root }, want, 64);
    defer none.deinit(gpa);
    try std.testing.expectEqual(calib_env.Source.measure, none.source);
    // a reference-engine entry for the shape: cached reads it, zig does not
    try tmp.dir.writeFile(io, .{ .sub_path = "calib-py.json", .data = "{\"verify\": [21.0, 26.0], \"draft\": 3.6, \"meta\": {\"shape\": {\"slots\": 4, \"world\": 2, \"dspark\": true}, \"time\": \"2026-10-07T12:00:00\"}}" });
    var verify = [_]f64{ 20.0004999, 24.5, 24.25, 30.0 };
    const c: costs_mod.Costs = .{ .verify = &verify, .draft = 3.5005, .slot = 0.611 };
    const kept = [_]f64{ 20.0, 24.4 };
    const path = try calib_env.store(gpa, io, root, want, c, .{ .method = .{}, .seconds = 7.25, .kept = &kept, .tokens = 201 }, 1_791_504_000);
    defer gpa.free(path);
    var nb: [64]u8 = undefined;
    const name = try calib_env.zigName(&nb, want);
    try std.testing.expect(std.mem.startsWith(u8, name, "calib-zig-") and name.len == "calib-zig-.json".len + 16);
    try std.testing.expectEqualStrings(name, std.fs.path.basename(path));
    var z = try calib_env.load(gpa, io, .{ .mode = "zig", .dir = root }, want, 64);
    defer z.deinit(gpa);
    try std.testing.expectEqual(calib_env.Source.cached, z.source);
    try std.testing.expectEqualStrings(name, std.fs.path.basename(z.path.?));
    try std.testing.expectEqualSlices(f64, &.{ 20.0, 24.5, 24.25, 30.0 }, z.costs.verify); // shared: microseconds
    try std.testing.expectEqual(@as(f64, 3.5), z.costs.draft); // 3500.5 us: half to even
    try std.testing.expectEqual(@as(f64, 0.611), z.costs.slot);
    var py = try calib_env.load(gpa, io, .{ .mode = "cached", .dir = root }, want, 64);
    defer py.deinit(gpa);
    try std.testing.expectEqualStrings("calib-py.json", std.fs.path.basename(py.path.?)); // never the Zig entry
    var real = try calib_env.load(gpa, io, .{ .mode = "real", .dir = root }, .{ .slots = 1, .world = 2, .dspark = true }, 64);
    defer real.deinit(gpa);
    try std.testing.expectEqualStrings("calib-py.json", std.fs.path.basename(real.path.?));
    // the stored file: the reference keys, the engine, a UTC time
    const bytes = try tmp.dir.readFileAlloc(io, name, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    const F = struct { engine: []const u8, meta: struct { time: []const u8, seconds: f64, prompt_tokens: u64, shape: struct { slots: u32, context: ?u64 }, method: struct { stat: []const u8, shape: []const u8, reps: u32 } } };
    const f = try std.json.parseFromSlice(F, gpa, bytes, .{ .ignore_unknown_fields = true });
    defer f.deinit();
    try std.testing.expectEqualStrings("zig", f.value.engine);
    try std.testing.expectEqualStrings("2026-10-09T00:00:00", f.value.meta.time);
    try std.testing.expectEqual(@as(u64, 1048576), f.value.meta.shape.context.?);
    try std.testing.expectEqualStrings("median", f.value.meta.method.stat);
    // another shape's Zig entry is not ours
    var other = try calib_env.load(gpa, io, .{ .mode = "zig", .dir = root }, .{ .slots = 4, .world = 1, .dspark = true }, 64);
    defer other.deinit(gpa);
    try std.testing.expectEqual(calib_env.Source.measure, other.source);
}
