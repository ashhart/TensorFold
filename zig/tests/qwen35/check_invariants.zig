//! The exactness checks of `check`: windows, drafts, shared rounds, resumed prompts and replayed graphs against serial.

const std = @import("std");
const check_ctx = @import("check_ctx.zig");
const ids_file = @import("ids_file.zig");
const lanes_session = @import("lanes_session.zig");
const rows_run = @import("rows_run.zig");
const check_radix = @import("check_radix.zig");

const Ctx = check_ctx.Ctx;
const Session = lanes_session.Session;

/// Windows of 1 to 16 rows, streams with a rejected draft's partial keep, and rounds padded up to a bucket.
const cases = blk: {
    var list: [25]rows_run.Case = undefined;
    for (0..16) |i| list[i] = .{ .streams = 1, .n = i + 1, .keep = i + 1 };
    list[16] = .{ .streams = 4, .n = 4, .keep = 4 };
    list[17] = .{ .streams = 4, .n = 4, .keep = 2 };
    list[18] = .{ .streams = 4, .n = 2, .keep = 2 };
    list[19] = .{ .streams = 4, .n = 2, .keep = 1 };
    list[20] = .{ .streams = 1, .n = 1, .keep = 1, .pad = 4 };
    list[21] = .{ .streams = 1, .n = 3, .keep = 3, .pad = 8 };
    list[22] = .{ .streams = 3, .n = 3, .keep = 2, .pad = 16 };
    list[23] = .{ .streams = 4, .n = 4, .keep = 4, .pad = 32 };
    list[24] = .{ .streams = 2, .n = 4, .keep = 1, .pad = 64 };
    break :blk list;
};

pub fn run(c: *Ctx) void {
    rows(c);
    lanesSet(c, "short", c.short);
    lanesSet(c, "long", c.long);
    resumed(c);
    check_radix.run(c);
    replayShare(c);
    check_radix.leaked(c);
}

fn rows(c: *Ctx) void {
    if (c.tp()) return c.report.skip("rows", "the layer by layer trace runs on one rank; the lane checks below run the windows under tp");
    const ids = c.ids[0..@min(c.ids.len, 256)];
    for (cases) |k| {
        const plain = std.fmt.allocPrint(c.arena, "rows {d} stream{s} x {d} row{s}, {d} kept", .{ k.streams, if (k.streams == 1) "" else "s", k.n, if (k.n == 1) "" else "s", k.keep }) catch return;
        const name = if (k.pad > 0) std.fmt.allocPrint(c.arena, "{s}, padded to {d}", .{ plain, k.pad }) catch return else plain;
        const diff = rows_run.runCase(c.gpa, c.e, ids, k) catch |err| {
            c.report.broke(name, err);
            continue;
        };
        if (diff) |d| {
            c.report.fail(name, "layer {d} ({s}), stream {d} row {d}: first difference at column {d}", .{ d.layer, d.kind, d.stream, d.row, d.column });
        } else c.report.pass(name, "every layer's residuals and the final rows equal each stream one row at a time", .{});
    }
}

/// Where two runs' replies first differ, as a line to print; null when they are equal.
fn differ(a: Session.Done, b: Session.Done, buf: []u8) ?[]const u8 {
    for (a.replies, b.replies) |x, y| {
        const at = std.mem.indexOfDiff(u32, x.tokens, y.tokens) orelse continue;
        const xt: i64 = if (at < x.tokens.len) x.tokens[at] else -1;
        const yt: i64 = if (at < y.tokens.len) y.tokens[at] else -1;
        return std.fmt.bufPrint(buf, "{s} differs at token {d}: {d} against {d}", .{ x.name, at, xt, yt }) catch "differs";
    }
    return null;
}

fn verdict(c: *Ctx, name: []const u8, a: Session.Done, b: Session.Done) void {
    var buf: [160]u8 = undefined;
    if (differ(a, b, &buf)) |why| return c.report.fail(name, "{s}", .{why});
    c.report.pass(name, "{d} streams, {d} tokens, {d} drafts accepted", .{ a.replies.len, a.tokens(), a.accepted() });
}

fn named(c: *Ctx, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.allocPrint(c.arena, fmt, args) catch "lanes";
}

/// Drafted against serial, solo against together and graph against eager, for each way of drawing.
fn lanesSet(c: *Ctx, set: []const u8, prompts: []const ids_file.Prompt) void {
    for (check_ctx.draws) |d| {
        const job: Session.Job = .{ .prompts = prompts, .max_new = c.tokens, .sampling = d.sampling };
        const together = c.session.run(c.arena, job) catch |err| return c.report.broke(named(c, "lanes {s} {s}", .{ set, d.name }), err);
        var serial = job;
        serial.drafts = false;
        const drafted = named(c, "lanes {s} {s}: drafted == serial", .{ set, d.name });
        if (c.session.h.head == null) {
            c.report.skip(drafted, "the model has no draft head");
        } else if (c.session.run(c.arena, serial)) |done| verdict(c, drafted, together, done) else |err| c.report.broke(drafted, err);
        var solo = job;
        solo.solo = true;
        if (c.session.run(c.arena, solo)) |done| verdict(c, named(c, "lanes {s} {s}: solo == together", .{ set, d.name }), together, done) else |err| c.report.broke(named(c, "lanes {s} {s}: solo == together", .{ set, d.name }), err);
        graphs(c, named(c, "lanes {s} {s}: graph == eager", .{ set, d.name }), job, together);
    }
}

/// `graph` ran with the engine's graphs; the same job again with every round eager.
fn graphs(c: *Ctx, name: []const u8, job: Session.Job, graph: Session.Done) void {
    if (c.tp()) return c.report.skip(name, "rounds under tp run eager unless --policy graphs=on");
    if (!c.e.o.graphs) return c.report.skip(name, "graphs are off");
    if (graph.replayed == 0) return c.report.skip(name, "no round of the run replayed a graph");
    c.e.o.graphs = false;
    defer c.e.o.graphs = true;
    const eager = c.session.run(c.arena, job) catch |err| return c.report.broke(name, err);
    var buf: [160]u8 = undefined;
    if (differ(graph, eager, &buf)) |why| return c.report.fail(name, "{s}", .{why});
    c.report.pass(name, "{d} rounds replayed, {d} tokens", .{ graph.replayed, graph.tokens() });
}

/// Each long prompt again, extended by its reply, on the caches the first run kept, against a run with none kept.
fn resumed(c: *Ctx) void {
    const name = "lanes long greedy: resumed == fresh";
    c.session.h.keepPrompts(8, 4 << 30);
    defer c.session.h.keepPrompts(0, 0);
    const first = c.session.run(c.arena, .{ .prompts = c.long, .max_new = c.tokens }) catch |err| return c.report.broke(name, err);
    const extended = lanes_session.extend(c.arena, c.long, first.replies) catch |err| return c.report.broke(name, err);
    const again = c.session.run(c.arena, .{ .prompts = extended, .max_new = c.tokens, .solo = true }) catch |err| return c.report.broke(name, err);
    const fresh = c.session.run(c.arena, .{ .prompts = extended, .max_new = c.tokens, .solo = true, .drafts = false }) catch |err| return c.report.broke(name, err);
    var cached: usize = 0;
    var every = true;
    for (again.replies) |r| {
        cached += r.cached;
        every = every and r.cached > 0;
    }
    var buf: [160]u8 = undefined;
    if (differ(again, fresh, &buf)) |why| return c.report.fail(name, "{s}", .{why});
    if (!every) return c.report.fail(name, "a prompt resumed from no kept caches ({d} tokens kept in all)", .{cached});
    c.report.pass(name, "{d} streams, {d} tokens, {d} prompt tokens from the kept caches", .{ again.replies.len, again.tokens(), cached });
}

/// A drafted run of every stream at once, the third time: once its shapes are met, rounds replay graphs.
fn replayShare(c: *Ctx) void {
    const name = "lanes short greedy: rounds replayed";
    if (c.tp() and !c.e.o.graphs) return c.report.skip(name, "rounds under tp run eager unless the policy says graphs=on");
    if (!c.e.o.graphs) return c.report.skip(name, "graphs are off");
    const job: Session.Job = .{ .prompts = c.short, .max_new = c.tokens };
    var done: Session.Done = undefined;
    for (0..3) |_| done = c.session.run(c.arena, job) catch |err| return c.report.broke(name, err);
    const share = 100.0 * @as(f64, @floatFromInt(done.replayed)) / @as(f64, @floatFromInt(@max(done.rounds, 1)));
    if (share < 99.0) return c.report.fail(name, "{d:.1}% of {d} rounds replayed", .{ share, done.rounds });
    c.report.pass(name, "{d:.1}% of {d} rounds replayed, {d:.0} us a round submitting", .{ share, done.rounds, @as(f64, @floatFromInt(done.submit_ns)) / @as(f64, @floatFromInt(@max(done.rounds, 1))) / 1e3 });
}
