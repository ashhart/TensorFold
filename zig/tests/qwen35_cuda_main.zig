//! `tf-qwen35-cuda <model dir> <step|window|tree|prompt> <ids> <out>`: the ids' logits as bf16 rows for a reference.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const E = q.engine;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 5) {
        std.debug.print("usage: tf-qwen35-cuda <model dir> <step|window|tree|prompt> <ids> <out>\n", .{});
        return 2;
    }
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, args[3], ", ");
    while (it.next()) |t| try ids.append(a, try std.fmt.parseInt(u32, t, 10));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const e = try E.Engine.init(init.gpa, init.io, &ctx, args[1]);
    defer e.deinit();
    // TF_Q35_PREFILL=N starts the stream after N cycled ids; TF_Q35_BESIDE=N steps an N-token stream beside it
    const prefill = try envCount("TF_Q35_PREFILL");
    const beside = try envCount("TF_Q35_BESIDE");
    var cache = try E.Cache.init(init.gpa, e, 4096 + prefill);
    defer cache.deinit(init.gpa);
    try fill(e, &cache, ids.items, prefill);
    var other = try E.Cache.init(init.gpa, e, 4096 + beside);
    defer other.deinit(init.gpa);
    try fill(e, &other, ids.items, beside);
    var f: Fwd = .{ .e = e, .other = if (beside > 0) &other else null, .ids = ids.items };
    var out: std.ArrayList(u8) = .empty;
    if (std.mem.eql(u8, args[2], "step")) {
        var first: i64 = 0;
        var rest: i64 = 0;
        for (ids.items, 0..) |id, i| {
            const t0 = std.Io.Clock.awake.now(init.io).toNanoseconds();
            try f.run(.{ .cache = &cache, .rows = 1 }, &.{id});
            const dt: i64 = @intCast(std.Io.Clock.awake.now(init.io).toNanoseconds() - t0);
            if (i == 0) first = dt else rest += dt;
            @memcpy(cache.logits, e.rowLogits(0));
            try out.appendSlice(a, std.mem.sliceAsBytes(cache.logits));
        }
        const n: f64 = @floatFromInt(ids.items.len - 1);
        std.debug.print("{d} steps: the first {d:.1} ms, then {d:.1} ms a step ({d:.1} tok/s)\n", .{ ids.items.len, @as(f64, @floatFromInt(first)) / 1e6, @as(f64, @floatFromInt(rest)) / 1e6 / n, n * 1e9 / @as(f64, @floatFromInt(rest)) });
    } else if (std.mem.eql(u8, args[2], "window")) {
        // windows of 1, 2, 3, ... rows: every row's logits, which must equal step's bit for bit
        var at: usize = 0;
        var w: usize = 1;
        while (at < ids.items.len) : (w = w % e.window_rows + 1) {
            const n = @min(w, ids.items.len - at);
            try f.run(.{ .cache = &cache, .rows = n }, ids.items[at..][0..n]);
            for (0..n) |r| try out.appendSlice(a, std.mem.sliceAsBytes(e.rowLogits(r)));
            at += n;
        }
    } else if (std.mem.eql(u8, args[2], "tree")) {
        // tree windows of growing widths, junk branches beside the kept true one, whose logits must equal step's bits
        var at: usize = 0;
        var w: usize = 2;
        while (at < ids.items.len) : (w = w % e.window_rows + 1) {
            var toks: [E.most_window]u32 = undefined;
            var parents: [E.most_window]i32 = undefined;
            var path: [E.most_window]u32 = undefined;
            toks[0] = ids.items[at];
            parents[0] = -1;
            path[0] = 0;
            var n: usize = 1;
            var kept: usize = 1;
            var last: i32 = 0;
            var depth: usize = 1;
            while (n + 2 <= w and at + depth < ids.items.len) : (depth += 1) {
                toks[n] = (ids.items[at + depth] + 7) % @as(u32, @intCast(e.g.vocab));
                parents[n] = last;
                toks[n + 1] = ids.items[at + depth];
                parents[n + 1] = last;
                path[kept] = @intCast(n + 1);
                last = @intCast(n + 1);
                n += 2;
                kept += 1;
            }
            if (n < w and n >= 3) {
                toks[n] = 1;
                parents[n] = 1;
                n += 1;
            }
            try f.run(.{ .cache = &cache, .rows = n, .parents = parents[0..n] }, toks[0..n]);
            for (path[0..kept]) |r| try out.appendSlice(a, std.mem.sliceAsBytes(e.rowLogits(r)));
            if (n > 1) try cache.keep(e, path[0..kept]);
            at += kept;
        }
    } else {
        var at: usize = 0;
        while (at < ids.items.len) {
            const n = @min(E.max_rows, ids.items.len - at);
            try e.forward(&.{.{ .cache = &cache, .rows = n }}, ids.items[at..][0..n], .last);
            at += n;
        }
        try out.appendSlice(a, std.mem.sliceAsBytes(cache.logits));
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = out.items });
    return 0;
}

/// One forward keeping logits: the stream's part first (its rows' logits come first), then the second stream's step.
const Fwd = struct {
    e: *E.Engine,
    other: ?*E.Cache,
    ids: []const u32,
    n: usize = 0,

    fn run(f: *Fwd, part: E.Part, toks: []const u32) !void {
        const o = f.other orelse return f.e.forward(&.{part}, toks, .logits);
        var all: [E.most_window + 1]u32 = undefined;
        @memcpy(all[0..toks.len], toks);
        all[toks.len] = f.ids[f.n % f.ids.len];
        f.n += 1;
        try f.e.forward(&.{ part, .{ .cache = o, .rows = 1 } }, all[0 .. toks.len + 1], .logits);
    }
};

fn envCount(name: [:0]const u8) !usize {
    const v = std.c.getenv(name) orelse return 0;
    return std.fmt.parseInt(usize, std.mem.span(v), 10);
}

/// `n` positions of the ids cycled, through prompt forwards.
fn fill(e: *E.Engine, cache: *E.Cache, ids: []const u32, n: usize) !void {
    var buf: [E.max_rows]u32 = undefined;
    var at: usize = 0;
    while (at < n) {
        const k = @min(E.max_rows, n - at);
        for (buf[0..k], at..) |*t, i| t.* = ids[i % ids.len];
        try e.forward(&.{.{ .cache = cache, .rows = k }}, buf[0..k], .last);
        at += k;
    }
}
