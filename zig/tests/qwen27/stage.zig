//! `tf-qwen27-stage <model> <out>`: one scenario's logits from one 27B engine, or rank N of a stage ring (TF_STAGE_*).
const std = @import("std");
const mtl = @import("metal");
const stage = @import("stage");
const S = @import("qwen27_stage");

const parallel = 2;
const context = 4096;
const chunk = 128;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) {
        std.debug.print("usage: tf-qwen27-stage <model> <out>; TF_STAGE_RANK, TF_STAGE_LAYOUT (first:end,...) for a ring\n", .{});
        return 2;
    }
    const dir = args[1];
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const config = try std.Io.Dir.cwd().readFileAlloc(init.io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 22));
    if (std.c.getenv("TF_STAGE_LOOP")) |end_text| { // layers [0, end) on one engine, the boundary out and back
        const end = try std.fmt.parseInt(usize, std.mem.span(end_text), 10);
        const e = S.Engine.init(init.gpa, init.io, dir, parallel, context, .{ .first = 0, .end = end, .heads = true }) catch |err| {
            std.debug.print("engine init failed: {s}\n", .{@errorName(err)});
            if (@errorReturnTrace()) |t| std.debug.dumpStackTrace(t);
            std.process.exit(1);
        };
        defer e.deinit();
        var loop: Loop = .{};
        var lead = try S.Lead.init(init.gpa, e, null, parallel, context, 1);
        defer lead.deinit(init.gpa);
        e.loopback = loop.hop();
        return finish(init, a, e, &lead, args[2], "loop");
    }
    if (std.c.getenv("TF_STAGE_RANK") == null) {
        const e = try S.Engine.init(init.gpa, init.io, dir, parallel, context, null);
        defer e.deinit();
        var lead = try S.Lead.init(init.gpa, e, null, parallel, context, 1);
        defer lead.deinit(init.gpa);
        return finish(init, a, e, &lead, args[2], "whole");
    }
    var set: stage.config.Settings = undefined;
    try stage.config.parse(stage.config.Env{}, &set);
    var ranges: [stage.config.most_ranks]stage.boot.Range = undefined;
    const layout = try parseLayout(std.mem.span(std.c.getenv("TF_STAGE_LAYOUT") orelse return error.MissingStageLayout), &ranges);
    if (layout.len != set.world) return error.BadStageLayout;
    const mine = layout[set.rank];
    const e = try S.Engine.init(init.gpa, init.io, dir, parallel, context, .{ .first = mine.first, .end = mine.end, .heads = set.rank == 0 });
    defer e.deinit();
    std.debug.print("stage {d}/{d} layers [{d},{d})\n", .{ set.rank, set.world, mine.first, mine.end });
    var ring = try stage.Ring.open(set.options(S.fingerprint(config, layout)));
    defer ring.close();
    defer ring.stop();
    const own = try S.describe(e, set.rank, set.world, mine, parallel, context);
    const model: stage.boot.Model = .{ .version = S.version, .kv_dtype = own.kv_dtype, .layers = @intCast(e.model.config.layers), .boundary = own.boundary, .shapes = own.shapes, .taps = &.{} };
    const next = try S.bootRing(init.gpa, &ring, &own, &model, layout);
    if (set.rank != 0) {
        try S.serve(init.gpa, e, &ring, parallel, context, next);
        std.debug.print("stage {d} served\n", .{set.rank});
        return 0;
    }
    var lead = try S.Lead.init(init.gpa, e, &ring, parallel, context, next);
    defer lead.deinit(init.gpa);
    return finish(init, a, e, &lead, args[2], "rank 0");
}

/// A ring of one: the rows given come back as the rows taken.
const Loop = struct {
    buf: [2 * 128 * 5120 * 2]u8 = undefined,
    n: usize = 0,
    fn hop(l: *Loop) @import("stage").driver.Hop {
        return .{ .ctx = l, .give = give, .take = take };
    }
    fn give(ctx: *anyopaque, _: usize, bytes: []const u8) anyerror!void {
        const l: *Loop = @ptrCast(@alignCast(ctx));
        @memcpy(l.buf[0..bytes.len], bytes);
        l.n = bytes.len;
    }
    fn take(ctx: *anyopaque, _: usize, bytes: []u8) anyerror!void {
        const l: *Loop = @ptrCast(@alignCast(ctx));
        if (bytes.len != l.n) return error.OtherRowCount;
        @memcpy(bytes, l.buf[0..l.n]);
    }
};

/// `first:end` ranges, comma-separated, in rank order.
fn parseLayout(text: []const u8, out: []stage.boot.Range) ![]const stage.boot.Range {
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

fn finish(init: std.process.Init, a: std.mem.Allocator, e: *S.Engine, lead: *S.Lead, out_path: []const u8, role: []const u8) !u8 {
    var out: std.ArrayList(u8) = .empty;
    const t0 = std.Io.Clock.awake.now(init.io).toNanoseconds();
    const toks = scenario(a, e, lead, &out) catch |err| {
        std.debug.print("stage {s}: scenario failed: {s}\n", .{ role, @errorName(err) });
        if (@errorReturnTrace()) |t| std.debug.dumpStackTrace(t);
        std.process.exit(1);
    };
    const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).toNanoseconds() - t0)) / 1e6;
    std.debug.print("stage {s}: scenario {d:.0} ms, {d} logits rows, tokens", .{ role, ms, out.items.len / (e.model.config.vocab * 2) });
    for (toks) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = out.items });
    return 0;
}

fn keepRow(a: std.mem.Allocator, e: *S.Engine, out: *std.ArrayList(u8), row: usize) !u32 {
    const logits = e.rowLogits(row);
    try out.appendSlice(a, std.mem.sliceAsBytes(logits));
    return argmax(logits);
}

/// A prompt in 128-row chunks, each a round; the last chunk's logits kept.
fn prompt(a: std.mem.Allocator, e: *S.Engine, lead: *S.Lead, out: *std.ArrayList(u8), id: u32, ids: []const u32) !u32 {
    var at: usize = 0;
    while (at < ids.len) {
        const n = @min(chunk, ids.len - at);
        try lead.step(&.{.{ .id = id, .op = if (at == 0) .open else .@"continue", .rows = @intCast(n) }}, ids[at..][0..n], .prompt);
        at += n;
    }
    return keepRow(a, e, out, 0);
}

/// The rounds both runs take, as qwen35_stage's on CUDA: every kept logits row appended; stream 1's greedy tokens.
fn scenario(a: std.mem.Allocator, e: *S.Engine, lead: *S.Lead, out: *std.ArrayList(u8)) ![]const u32 {
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const first = try a.alloc(u32, 600); // four whole chunks and a ragged one
    for (first) |*t| t.* = rnd.intRangeLessThan(u32, 1000, 60000);
    const other = try a.alloc(u32, 300);
    for (other) |*t| t.* = rnd.intRangeLessThan(u32, 1000, 60000);
    var toks: std.ArrayList(u32) = .empty;
    var t1 = try prompt(a, e, lead, out, 1, first);
    try toks.append(a, t1);
    var kept1: u32 = 0;
    for (0..8) |_| {
        try lead.step(&.{.{ .id = 1, .op = .@"continue", .rows = 1, .accepted = kept1 }}, &.{t1}, .window);
        t1 = try keepRow(a, e, out, 0);
        try toks.append(a, t1);
        kept1 = 1;
    }
    var t2 = try prompt(a, e, lead, out, 2, other);
    var kept2: u32 = 0;
    for (0..8) |_| {
        const both = [_]S.Entry{ .{ .id = 1, .op = .@"continue", .rows = 1, .accepted = kept1 }, .{ .id = 2, .op = .@"continue", .rows = 1, .accepted = kept2 } };
        try lead.step(&both, &.{ t1, t2 }, .window);
        t1 = try keepRow(a, e, out, 0);
        t2 = try keepRow(a, e, out, 1);
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
        for (0..5) |r| _ = try keepRow(a, e, out, r);
        t1 = argmax(e.rowLogits(take - 1));
        try toks.append(a, t1);
        kept1 = if (w == 2) 0 else take; // window 2 rolls back whole: its first row runs again
        std.debug.print("window {d}: {d} of 5 rows kept\n", .{ w, kept1 });
        if (w == 2) t1 = win[0];
    }
    try lead.step(&.{.{ .id = 1, .op = .@"continue", .rows = 1, .accepted = kept1 }}, &.{t1}, .window);
    try toks.append(a, try keepRow(a, e, out, 0));
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
