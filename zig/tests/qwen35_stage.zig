//! `tf-qwen35-cuda <model> stage <out> [fp8]`: one scenario's logits from one engine, or rank 0 of a stage ring.
const std = @import("std");
const cuda = @import("cuda");
const stage = @import("stage");
const q = @import("qwen35");
const E = q.engine;
const S = q.stage;

const parallel = 2;
const context = 8192;

pub fn run(init: std.process.Init, a: std.mem.Allocator, dir: []const u8, out_path: []const u8, kv8: bool) !u8 {
    const config = try std.Io.Dir.cwd().readFileAlloc(init.io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 22));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    if (std.c.getenv("TF_STAGE_RANK") == null) {
        const e = try E.Engine.init(init.gpa, init.io, &ctx, dir, kv8, null, E.max_rows);
        defer e.deinit();
        var lead = try S.Lead.init(init.gpa, e, null, parallel, context, 1);
        defer lead.deinit(init.gpa);
        return finish(init, a, e, &lead, out_path, "whole");
    }
    var set: stage.config.Settings = undefined;
    try stage.config.parse(stage.config.Env{}, &set);
    var ranges: [stage.config.most_ranks]stage.boot.Range = undefined;
    const layout = try parseLayout(std.mem.span(std.c.getenv("TF_STAGE_LAYOUT") orelse return error.MissingStageLayout), &ranges);
    if (layout.len != set.world) return error.BadStageLayout;
    const mine = layout[set.rank];
    const e = try E.Engine.initStage(init.gpa, init.io, &ctx, dir, kv8, null, E.max_rows, .{ .first = mine.first, .end = mine.end, .heads = set.rank == 0 });
    defer e.deinit();
    const loaded = cuda.usage(false);
    std.debug.print("stage {d}/{d} layers [{d},{d}): {d:.2} GiB device, {d:.2} GiB pinned after load\n", .{ set.rank, set.world, mine.first, mine.end, gib(loaded.device), gib(loaded.host) });
    var ring = try stage.Ring.open(set.options(S.fingerprint(config, layout)));
    defer ring.close();
    defer ring.stop();
    const own = try S.describe(e, set.rank, set.world, mine, parallel, context);
    const model: stage.boot.Model = .{ .version = S.version, .kv_dtype = own.kv_dtype, .layers = @intCast(e.g.model().layers), .boundary = own.boundary, .shapes = own.shapes, .taps = &.{} };
    const next = try S.bootRing(init.gpa, &ring, &own, &model, layout);
    if (set.rank != 0) {
        try S.serve(init.gpa, e, &ring, parallel, context, next);
        report(set.rank, "served");
        return 0;
    }
    var lead = try S.Lead.init(init.gpa, e, &ring, parallel, context, next);
    defer lead.deinit(init.gpa);
    return finish(init, a, e, &lead, out_path, "rank 0");
}

fn gib(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / (1 << 30);
}

pub fn report(rank: u32, what: []const u8) void {
    const u = cuda.usage(false);
    std.debug.print("stage {d} {s}: {d:.2} GiB device now, {d:.2} GiB peak\n", .{ rank, what, gib(u.device), gib(u.peak) });
}

/// `first:end` ranges, comma-separated, in rank order.
pub fn parseLayout(text: []const u8, out: []stage.boot.Range) ![]const stage.boot.Range {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |r| {
        if (n == out.len) return error.BadStageLayout;
        const colon = std.mem.indexOfScalar(u8, r, ':') orelse return error.BadStageLayout;
        out[n] = .{ .first = try std.fmt.parseInt(u32, r[0..colon], 10), .end = try std.fmt.parseInt(u32, r[colon + 1 ..], 10) };
        n += 1;
    }
    return out[0..n];
}

fn finish(init: std.process.Init, a: std.mem.Allocator, e: *E.Engine, lead: *S.Lead, out_path: []const u8, role: []const u8) !u8 {
    var out: std.ArrayList(u8) = .empty;
    const t0 = std.Io.Clock.awake.now(init.io).toNanoseconds();
    const toks = try scenario(a, e, lead, &out);
    const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).toNanoseconds() - t0)) / 1e6;
    std.debug.print("stage {s}: scenario {d:.0} ms, {d} logits rows, tokens", .{ role, ms, out.items.len / (e.g.vocab * 2) });
    for (toks) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});
    report(0, role);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = out.items });
    return 0;
}

/// The rounds both runs take; every kept logits row appended to `out`; the greedy tokens of stream 1 returned.
fn scenario(a: std.mem.Allocator, e: *E.Engine, lead: *S.Lead, out: *std.ArrayList(u8)) ![]const u32 {
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const prompt = try a.alloc(u32, 4096 + 904);
    for (prompt) |*t| t.* = rnd.intRangeLessThan(u32, 1000, 60000);
    const other = try a.alloc(u32, 300);
    for (other) |*t| t.* = rnd.intRangeLessThan(u32, 1000, 60000);
    var toks: std.ArrayList(u32) = .empty;
    // stream 1's prompt in two chunks, then stream 2's beside it
    try lead.step(&.{.{ .id = 1, .op = .open, .rows = 4096 }}, prompt[0..4096], .prompt);
    try out.appendSlice(a, std.mem.sliceAsBytes(lead.cache(1).?.logits));
    try lead.step(&.{.{ .id = 1, .op = .@"continue", .rows = 904 }}, prompt[4096..], .prompt);
    try out.appendSlice(a, std.mem.sliceAsBytes(lead.cache(1).?.logits));
    var t1 = argmax(lead.cache(1).?.logits);
    try toks.append(a, t1);
    var kept1: u32 = 0; // stream 1's previous window: no window after a prompt chunk
    for (0..8) |_| {
        try lead.step(&.{.{ .id = 1, .op = .@"continue", .rows = 1, .accepted = kept1 }}, &.{t1}, .window);
        try out.appendSlice(a, std.mem.sliceAsBytes(e.rowLogits(0)));
        t1 = argmax(e.rowLogits(0));
        try toks.append(a, t1);
        kept1 = 1;
    }
    try lead.step(&.{.{ .id = 2, .op = .open, .rows = 300 }}, other, .prompt);
    try out.appendSlice(a, std.mem.sliceAsBytes(lead.cache(2).?.logits));
    var t2 = argmax(lead.cache(2).?.logits);
    var kept2: u32 = 0;
    for (0..8) |_| {
        const both = [_]S.Entry{ .{ .id = 1, .op = .@"continue", .rows = 1, .accepted = kept1 }, .{ .id = 2, .op = .@"continue", .rows = 1, .accepted = kept2 } };
        try lead.step(&both, &.{ t1, t2 }, .window);
        for (0..2) |r| try out.appendSlice(a, std.mem.sliceAsBytes(e.rowLogits(r)));
        t1 = argmax(e.rowLogits(0));
        t2 = argmax(e.rowLogits(1));
        try toks.append(a, t1);
        kept2 = 1;
    }
    try lead.beat();
    // drafted windows on stream 1: its next token, repeats of it as drafts up to a cut, then junk
    const cut = [_]usize{ 3, 5, 3, 2 }; // a greedy check keeps a prefix of 1 to 5 rows; window 2 rolls back whole
    for (0..4) |w| {
        var win: [5]u32 = @splat(t1);
        for (win[cut[w]..]) |*d| d.* = rnd.intRangeLessThan(u32, 1000, 60000);
        try lead.step(&.{.{ .id = 1, .op = .@"continue", .rows = 5, .accepted = kept1 }}, &win, .window);
        var take: u32 = 1;
        while (take < 5 and argmax(e.rowLogits(take - 1)) == win[take]) take += 1;
        for (0..5) |r| try out.appendSlice(a, std.mem.sliceAsBytes(e.rowLogits(r)));
        t1 = argmax(e.rowLogits(take - 1));
        try toks.append(a, t1);
        kept1 = if (w == 2) 0 else take; // window 2 rolls back whole: its first row runs again
        std.debug.print("window {d}: {d} of 5 rows kept\n", .{ w, kept1 });
        if (w == 2) t1 = win[0];
    }
    try lead.step(&.{.{ .id = 1, .op = .@"continue", .rows = 1, .accepted = kept1 }}, &.{t1}, .window);
    try out.appendSlice(a, std.mem.sliceAsBytes(e.rowLogits(0)));
    try toks.append(a, argmax(e.rowLogits(0)));
    try lead.step(&.{ .{ .id = 1, .op = .close }, .{ .id = 2, .op = .close } }, &.{}, .window);
    return toks.items;
}

fn argmax(row: []const u16) u32 {
    var best: usize = 0;
    var bv: f32 = -std.math.inf(f32);
    for (row, 0..) |h, i| {
        const v: f32 = @bitCast(@as(u32, h) << 16);
        if (v > bv) {
            bv = v;
            best = i;
        }
    }
    return @intCast(best);
}
