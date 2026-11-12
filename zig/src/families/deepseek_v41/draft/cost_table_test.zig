//! The depth rule priced from a measured table: the defaults' timings give today's choices, a launch step is avoided.
const std = @import("std");
const cm = @import("calib_measure.zig");
const costs_mod = @import("costs.zig");
const depth = @import("depth.zig");
const calib_env = @import("calib_env.zig");
const plain = @import("plain_streams_test.zig");

const gpa = std.testing.allocator;
const t_ = std.testing;

/// A table with a flat price a row and a second launch's price from row 17 on: `step` ms more.
fn launchTable(buf: *[32]f64, base: f64, row: f64, step: f64) costs_mod.Costs {
    for (buf, 1..) |*v, r| v.* = base + row * @as(f64, @floatFromInt(r - 1)) + (if (r > 16) step else 0.0);
    return .{ .verify = buf, .draft = 12.0, .slot = 0.5 };
}

/// The rounds' total rows when `n` streams each ask for five drafts of confidence `p`.
fn roundRows(c: costs_mod.Costs, set: depth.Settings, n: usize, p: f64) !usize {
    var d = try depth.Depth.init(gpa, c, 5, set);
    defer d.deinit();
    const conf = [5]f64{ p, p, p, p, p };
    var asks: [4]depth.Depth.Ask = undefined;
    for (asks[0..n], 0..) |*a, i| a.* = .{ .slot = @intCast(i), .conf = &conf, .most = 5, .first = 7, .position = 100 };
    var ks: [4]usize = undefined;
    try d.chooseJoint(asks[0..n], 0, 0, null, ks[0..n]);
    var rows: usize = 0;
    for (ks[0..n]) |k| rows += 1 + k;
    return rows;
}

test "a table timed at the defaults' own values is the defaults' table, and the rule chooses as it did" {
    var p = try cm.Plan.initWide(gpa, 4, 64, 16, cm.launch_bound, true, &cm.launch_wide);
    defer p.deinit(gpa);
    try t_.expectEqualSlices(u32, &cm.launch_wide, p.wideBucketsOf());
    var lin = try costs_mod.defaults(gpa, 64);
    defer lin.deinit(gpa);
    // [pass, rows 1..16, the 4 + 4 window, the wide windows]
    var rank: [1 + 16 + 1 + 2]f64 = undefined;
    rank[0] = lin.draft;
    for (rank[1..17], 1..) |*x, r| x.* = lin.windowMs(@intCast(r));
    rank[17] = lin.windowMs(8);
    for (rank[18..], cm.launch_wide) |*x, b| x.* = lin.windowMs(b);
    try t_.expectEqual(p.entries(), rank.len);
    var built = try p.build(gpa, &.{&rank});
    defer built.deinit(gpa);
    try t_.expectEqual(@as(usize, 32), built.verify.len);
    try t_.expectEqualSlices(f64, lin.verify[0..32], built.verify);
    try t_.expectEqual(lin.draft, built.draft);
    try t_.expectEqual(@as(f64, 0.0), built.slot);
    for ([_]u2{ 0, 1, 2 }) |joint| for ([_]f64{ 0.95, 0.8, 0.6, 0.4 }) |conf| for (1..5) |n| {
        const set: depth.Settings = .{ .joint = joint };
        try t_.expectEqual(try roundRows(lin, set, n, conf), try roundRows(built, set, n, conf));
    };
}

test "a step at the second launch keeps a round at one launch when the rows do not pay for it" {
    var buf: [32]f64 = undefined;
    const step = launchTable(&buf, 28.0, 6.6, 28.0);
    var lin = try costs_mod.defaults(gpa, 64);
    defer lin.deinit(gpa);
    // confident drafts of 3 and 4 streams: the linear table pays for 18 and 24 rows, the step keeps them in 16
    for ([_]usize{ 3, 4 }) |n| {
        try t_.expect(try roundRows(lin, .{}, n, 0.95) > 16);
        try t_.expect(try roundRows(step, .{}, n, 0.95) <= 16);
    }
    // a lone stream's six rows and every round already inside 16 rows price the same either way
    try t_.expectEqual(try roundRows(lin, .{}, 1, 0.95), try roundRows(step, .{}, 1, 0.95));
    // the step is a price, not a cap: a small one is paid when the drafts are confident enough
    var cheap_buf: [32]f64 = undefined;
    const cheap = launchTable(&cheap_buf, 28.0, 6.6, 2.0);
    try t_.expect(try roundRows(cheap, .{}, 4, 0.95) > 16);
}

test "the switch still forces plain at its threshold over a table with a step, and below it the rule drafts" {
    var buf: [32]f64 = undefined;
    const step = launchTable(&buf, 28.0, 6.6, 28.0);
    const set: depth.Settings = .{ .plain_streams = 3 };
    const lens = [_]u32{ 40, 40, 40, 40 };
    const four = [_][]const u32{ &plain.a1, &plain.a1, &plain.a1, &plain.a1 };
    const at = try plain.run(&four, &lens, .{ .set = set, .costs = step });
    defer at.deinit();
    try t_.expectEqual(@as(u64, 0), at.drafted);
    try plain.expectSerial(at, &four, &lens, null);
    const below = try plain.run(four[0..2], lens[0..2], .{ .set = set, .costs = step });
    defer below.deinit();
    try t_.expect(below.drafted > 0);
    try plain.expectSerial(below, four[0..2], lens[0..2], null);
    const off = try plain.run(&four, &lens, .{ .costs = step });
    defer off.deinit();
    try t_.expect(off.drafted > 0);
    try plain.expectSerial(off, &four, &lens, null);
}

const Env = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: [*:0]const u8) ?[]const u8 {
        for (pairs) |kv| if (std.mem.eql(u8, kv[0], std.mem.span(name))) return kv[1];
        return null;
    }
};

test "the launch-bound method is the backend's default and every knob still overrides it; the reference's is unchanged" {
    Env.pairs = &.{};
    try t_.expectEqual(cm.launch_bound, try cm.methodFrom(&Env.get, cm.launch_bound));
    try t_.expectEqual(cm.Method{}, try cm.method(&Env.get));
    try t_.expectEqual(@as(u32, 0), (try cm.method(&Env.get)).budget_ms);
    Env.pairs = &.{ .{ "TF_DSV41_CALIB_BUDGET_MS", "0" }, .{ "TF_DSV41_CALIB_REPS", "7" }, .{ "TF_DSV41_CALIB_STAT", "median" } };
    defer Env.pairs = &.{};
    const m = try cm.methodFrom(&Env.get, cm.launch_bound);
    try t_.expectEqual(@as(u32, 0), m.budget_ms);
    try t_.expectEqual(@as(u32, 7), m.reps);
    try t_.expectEqual(cm.Stat.median, m.stat);
    try t_.expectEqual(@as(u32, 1), m.warm);
    Env.pairs = &.{.{ "TF_DSV41_CALIB_BUDGET_MS", "x" }};
    try t_.expectError(error.InvalidCharacter, cm.methodFrom(&Env.get, cm.launch_bound));
}

/// What a window of `rows` rows over `slots` slots costs on a launch-bound model: two launches past 16 rows.
fn synthetic(rows: u32, slots: u32) f64 {
    const launches: f64 = @floatFromInt((rows + 15) / 16);
    return 28.0 * launches + 6.6 * @as(f64, @floatFromInt(rows)) + 7.0 * @as(f64, @floatFromInt(slots - 1));
}

test "a sparse plan under a spent budget still gives a monotone 32-row table from a few timed windows" {
    var one = try cm.Plan.initWide(gpa, 1, 64, 16, cm.launch_bound, true, &cm.launch_wide);
    defer one.deinit(gpa);
    try t_.expectEqual(@as(usize, 0), one.wideBucketsOf().len);
    var p = try cm.Plan.initWide(gpa, 4, 64, 16, cm.launch_bound, true, &cm.launch_wide);
    defer p.deinit(gpa);
    try t_.expectEqualSlices(u32, &cm.launch_wide, p.wideBucketsOf());
    // every pass after the first is dropped: the budget was spent by the first
    const ms = try gpa.alloc(f64, p.runs.items.len);
    defer gpa.free(ms);
    var timed: usize = 0;
    var spent: f64 = 0.0;
    var pass_runs: usize = 0;
    for (p.runs.items, ms, 0..) |r, *y, i| {
        if (i > 0 and !r.warm and !r.must and p.repeat(i)) {
            y.* = -1.0;
            continue;
        }
        var rows: u32 = 0;
        for (r.segments()) |sg| rows += sg.rows;
        y.* = if (r.what == .draft) 12.0 else synthetic(rows, r.nsegs);
        spent += y.*;
        timed += 1;
        pass_runs += @intFromBool(!r.warm);
    }
    // 2 untimed, 9 rows, the 4 + 4 window, 2 wide windows and the pass once each, the slot term's pair twice more
    try t_.expectEqual(@as(usize, 2 + 9 + 1 + 2 + 1 + 4), timed);
    try t_.expectEqual(@as(usize, 13 + 4), pass_runs);
    var eights: usize = 0;
    var pairs: usize = 0;
    var passes: usize = 0;
    for (p.runs.items, ms) |r, y| {
        if (r.warm or y < 0) continue;
        eights += @intFromBool(r.what == .sweep and r.what.sweep == 8);
        pairs += @intFromBool(r.what == .shared and r.what.shared == 0);
        passes += @intFromBool(r.what == .draft);
    }
    try t_.expectEqual(@as(usize, 3), eights);
    try t_.expectEqual(@as(usize, 3), pairs);
    try t_.expectEqual(@as(usize, 1), passes); // the pass keeps its warm-up and runs inside, back to back
    try t_.expect(spent < 2000.0);
    const out = try gpa.alloc(f64, p.entries());
    defer gpa.free(out);
    try p.values(ms, out);
    var built = try p.build(gpa, &.{out});
    defer built.deinit(gpa);
    try t_.expectEqual(@as(usize, 32), built.verify.len);
    for (built.verify[1..], built.verify[0..31]) |b, a| try t_.expect(b >= a);
    // the timed rows are exact and the ones between them are on the line
    for ([_]u32{ 1, 2, 3, 4, 5, 6, 8, 12, 16 }) |r| try t_.expectApproxEqAbs(synthetic(r, 1), built.verify[r - 1], 0.002);
    try t_.expectApproxEqAbs((synthetic(6, 1) + synthetic(8, 1)) / 2.0, built.verify[6], 0.01); // row 7 is on the line
    try t_.expect(built.verify[16] - built.verify[15] > 20.0); // the second launch's step is in the table
    try t_.expectEqual(@as(f64, 12.0), built.draft);
}

test "a plan with budget left repeats its passes, and a repeat never replaces an entry's first timed run" {
    var p = try cm.Plan.initWide(gpa, 2, 64, 16, cm.launch_bound, true, &cm.launch_wide);
    defer p.deinit(gpa);
    var first: usize = 0;
    for (p.runs.items, 0..) |r, i| {
        var seen = false;
        for (p.runs.items[0..i]) |o| seen = seen or (!o.warm and std.meta.eql(o.what, r.what));
        try t_.expectEqual(seen, p.repeat(i));
        first += @intFromBool(!r.warm and !seen);
    }
    try t_.expectEqual(@as(usize, 13), first); // 9 rows, the 4 + 4 window, 2 wide windows, the pass
}

test "a stored table is read back only for the device and weights that stored it" {
    const io = t_.io;
    var tmp = t_.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [256]u8 = undefined;
    const root = try std.fmt.bufPrint(&buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const base: calib_env.Shape = .{ .slots = 4, .world = 1, .dspark = true, .context = 1280 };
    var one = [_]f64{ 30.0, 40.0 };
    var other = [_]f64{ 50.0, 60.0 };
    const mine: costs_mod.Costs = .{ .verify = &one, .draft = 10.0, .slot = 1.0 };
    const theirs: costs_mod.Costs = .{ .verify = &other, .draft = 20.0, .slot = 2.0 };
    var a = base;
    a.id = 0xa1;
    var b = base;
    b.id = 0xb2;
    const pa = try calib_env.store(gpa, io, root, a, mine, .{ .method = .{}, .seconds = 1.0 }, 1_791_504_000);
    defer gpa.free(pa);
    const p0 = try calib_env.store(gpa, io, root, base, theirs, .{ .method = .{}, .seconds = 1.0 }, 1_791_504_001);
    defer gpa.free(p0);
    try t_.expect(!std.mem.eql(u8, std.fs.path.basename(pa), std.fs.path.basename(p0)));
    var got = try calib_env.load(gpa, io, .{ .mode = "zig", .dir = root }, a, 64);
    defer got.deinit(gpa);
    try t_.expectEqual(calib_env.Source.cached, got.source);
    try t_.expectEqual(@as(f64, 10.0), got.costs.draft);
    var none = try calib_env.load(gpa, io, .{ .mode = "zig", .dir = root }, b, 64);
    defer none.deinit(gpa);
    try t_.expectEqual(calib_env.Source.measure, none.source);
    var unnamed = try calib_env.load(gpa, io, .{ .mode = "zig", .dir = root }, base, 64);
    defer unnamed.deinit(gpa);
    try t_.expectEqual(calib_env.Source.cached, unnamed.source);
    try t_.expectEqual(@as(f64, 20.0), unnamed.costs.draft);
}

/// Tokens a lone stream drafting `k` of chain chance `p` commits a round, on average.
fn committed(p: f64, k: usize) f64 {
    var run: f64 = 1.0;
    var e: f64 = 1.0;
    for (0..k) |_| {
        run *= p;
        e += run;
    }
    return e;
}

/// The drafts the rule gives a lone stream at chain chance `p`, once its rate has settled on its own rounds.
fn loneDrafts(c: costs_mod.Costs, p: f64) !usize {
    var d = try depth.Depth.init(gpa, c, 5, .{});
    defer d.deinit();
    const conf = [5]f64{ p, p, p, p, p };
    var k: usize = 0;
    for (0..40) |_| {
        const ask = [_]depth.Depth.Ask{.{ .slot = 0, .conf = &conf, .most = 5, .first = 7, .position = 100 }};
        var ks: [1]usize = undefined;
        try d.chooseJoint(&ask, 0, 0, true, &ks);
        k = ks[0];
        (try d.state(0)).rounds.push(.{ .tokens = committed(p, k), .ms = d.roundMs(k + 1, 1, true) });
    }
    return k;
}

test "a lone stream's choice under the measured table is never worse, under the measured costs, than the default table's" {
    const xs = [_]u32{ 1, 2, 3, 4, 5, 6, 8, 16 };
    const ys = [_]f64{ 28.4, 37.5, 44.2, 50.9, 58.75, 66.6, 82.3, 152.0 };
    var buf: [32]f64 = undefined;
    for (&buf, 1..) |*v, r| {
        var i: usize = 0;
        while (i + 2 < xs.len and xs[i + 1] < r) i += 1;
        const f = (@as(f64, @floatFromInt(r)) - @as(f64, @floatFromInt(xs[i]))) / @as(f64, @floatFromInt(xs[i + 1] - xs[i]));
        v.* = ys[i] + (ys[i + 1] - ys[i]) * f + (if (r > 16) @as(f64, 19.5) else 0.0);
    }
    const measured: costs_mod.Costs = .{ .verify = &buf, .draft = 6.42, .slot = 7.96 };
    var lin = try costs_mod.defaults(gpa, 64);
    defer lin.deinit(gpa);
    for ([_]f64{ 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 }) |p| {
        const km = try loneDrafts(measured, p);
        const kd = try loneDrafts(lin, p);
        // tokens a ms under the measured costs, the pass paid every round
        const got = committed(p, km) / (measured.windowMs(@intCast(km + 1)) + measured.draft);
        const base = committed(p, kd) / (measured.windowMs(@intCast(kd + 1)) + measured.draft);
        try t_.expect(got >= base - 1e-9);
        try t_.expect(km <= kd); // dearer rows past the first never draft deeper
    }
}

test "a stored table whose pass is implausible against its rows is never read back" {
    const io = t_.io;
    var tmp = t_.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [256]u8 = undefined;
    const root = try std.fmt.bufPrint(&buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const want: calib_env.Shape = .{ .slots = 4, .world = 1, .dspark = true, .context = 1280, .id = 0xa1 };
    var v = [_]f64{ 28.4, 37.5 };
    const bad: costs_mod.Costs = .{ .verify = &v, .draft = 105.0, .slot = 8.0 };
    const path = try calib_env.store(gpa, io, root, want, bad, .{ .method = .{}, .seconds = 1.0 }, 1_791_504_000);
    defer gpa.free(path);
    var got = try calib_env.load(gpa, io, .{ .mode = "zig", .dir = root }, want, 64);
    defer got.deinit(gpa);
    try t_.expectEqual(calib_env.Source.measure, got.source); // measured again, not the stored 105 ms
    try t_.expectError(error.ImplausibleCosts, calib_env.load(gpa, io, .{ .mode = "cached", .file = path }, want, 64));
}
