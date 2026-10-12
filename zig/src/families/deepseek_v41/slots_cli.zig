//! `tf-dsv41 slots PACK ASSETS TRACE` (rank 0; others `follow`): batched and alone replies equal TRACE, greedy.

const std = @import("std");
const api = @import("engine_api");
const se = @import("serve_engine.zig");

const Box = struct {
    io: std.Io,
    gpa: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?api.Reason = null,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        switch (e.*) {
            .tokens => |t| b.tokens.appendSlice(b.gpa, t) catch {},
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }

    fn finished(b: *Box) ?api.Reason {
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        return b.done;
    }
};

const Line = struct { prompt: []const u32, tokens: []const u32 };

/// `got` equals `want`, or `want` cut at its first `eos` with or without the id.
fn same(got: []const u32, want: []const u32, eos: u32) bool {
    if (std.mem.eql(u32, got, want)) return true;
    const cut = std.mem.indexOfScalar(u32, want, eos) orelse return false;
    return std.mem.eql(u32, got, want[0 .. cut + 1]) or std.mem.eql(u32, got, want[0..cut]);
}

fn firstDiff(got: []const u32, want: []const u32) i64 {
    return @intCast(std.mem.indexOfDiff(u32, got, want) orelse @min(got.len, want.len));
}

/// Submits lines[i] for each i in `which` at once and waits for all of them.
fn submitAll(eng: api.Engine, io: std.Io, lines: []const Line, which: []const usize, boxes: []Box, id0: api.Id, drafts: bool) !void {
    var reqs: [16]api.Request = undefined;
    if (which.len > reqs.len) return error.TooManyLines;
    for (which, 0..) |i, k| {
        reqs[k] = .{ .prompt = lines[i].prompt, .max_tokens = @intCast(lines[i].tokens.len), .drafts = drafts };
        try eng.submit(id0 + k, &reqs[k], .{ .ctx = &boxes[i], .event = Box.event });
    }
    while (true) {
        var all = true;
        for (which) |i| all = all and boxes[i].finished() != null;
        if (all) return;
        std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    }
}

pub fn main(gpa: std.mem.Allocator, io: std.Io, pack: []const u8, assets: []const u8, trace: []const u8) !u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, trace, a, .limited(1 << 30));
    var lines: std.ArrayList(Line) = .empty;
    var it = std.mem.tokenizeScalar(u8, text, '\n');
    while (it.next()) |l| try lines.append(a, try std.json.parseFromSliceLeaky(Line, a, l, .{ .ignore_unknown_fields = true }));
    const n = lines.items.len;
    if (n == 0 or n > 16) return error.BadTrace;
    const drafts = if (std.c.getenv("TF_DSV41_DRAFTS")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else false;
    const s = try se.Served.open(gpa, io, pack, assets, drafts, null);
    defer se.Served.close(s);
    const eng = s.engine();
    const eos = s.m.cfg.eos;
    std.debug.print("slots check: {d} live slots, {d} prompts, drafts {}\n", .{ s.m.slots, n, drafts });
    const batched = try a.alloc(Box, n);
    const alone = try a.alloc(Box, n);
    for (batched, alone) |*x, *y| {
        x.* = .{ .io = io, .gpa = a };
        y.* = .{ .io = io, .gpa = a };
    }
    const all = try a.alloc(usize, n);
    for (all, 0..) |*x, i| x.* = i;
    var t0 = std.Io.Clock.awake.now(io);
    try submitAll(eng, io, lines.items, all, batched, 1, drafts);
    const tb = t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    t0 = std.Io.Clock.awake.now(io);
    for (0..n) |i| try submitAll(eng, io, lines.items, &.{i}, alone, 100 + i, drafts);
    const ta = t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    var ok_b: usize = 0;
    var ok_a: usize = 0;
    var ok_ba: usize = 0;
    var toks: usize = 0;
    for (lines.items, batched, alone, 0..) |l, *b, *x, i| {
        const eb = same(b.tokens.items, l.tokens, eos);
        const ea = same(x.tokens.items, l.tokens, eos);
        const eba = std.mem.eql(u32, b.tokens.items, x.tokens.items);
        ok_b += @intFromBool(eb);
        ok_a += @intFromBool(ea);
        ok_ba += @intFromBool(eba);
        toks += b.tokens.items.len;
        std.debug.print("{{\"line\": {d}, \"prompt\": {d}, \"recorded\": {d}, \"batched\": {d}, \"alone\": {d}, \"batched_eq_ref\": {}, \"alone_eq_ref\": {}, \"batched_eq_alone\": {}, \"first_diff_batched\": {d}, \"first_diff_alone\": {d}, \"finished\": \"{t}\"}}\n", .{ i, l.prompt.len, l.tokens.len, b.tokens.items.len, x.tokens.items.len, eb, ea, eba, if (eb) -1 else firstDiff(b.tokens.items, l.tokens), if (ea) -1 else firstDiff(x.tokens.items, l.tokens), b.done.? });
    }
    const pass = ok_b == n and ok_a == n and ok_ba == n;
    std.debug.print("{s} slots batched == Python: {d}/{d}\n", .{ if (ok_b == n) "PASS" else "FAIL", ok_b, n });
    std.debug.print("{s} slots alone == Python: {d}/{d}\n", .{ if (ok_a == n) "PASS" else "FAIL", ok_a, n });
    std.debug.print("{s} slots batched == alone: {d}/{d}\n", .{ if (ok_ba == n) "PASS" else "FAIL", ok_ba, n });
    std.debug.print("slots timing: batched {d} ms, alone {d} ms, {d} tokens\n", .{ @divTrunc(tb, 1_000_000), @divTrunc(ta, 1_000_000), toks });
    if (s.m.rows) |r| std.debug.print("row windows: {d}, rows {d}, padded {d}\n", .{ r.stats.windows, r.stats.rows, r.stats.padded });
    return if (pass) 0 else 1;
}
