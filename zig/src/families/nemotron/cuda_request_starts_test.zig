//! Message-start cuts in Nemotron's prompt pass: a resume cuts at the next start, not the grid step.
const std = @import("std");
const pass = @import("cuda_prompt.zig");
const t = std.testing;
const lanes = @import("lanes");
const state = @import("cuda_state.zig");

/// Chunk ends a pass walks, as serialChunks does: each end is a cut where a state is kept.
fn walk(out: []usize, from: usize, len: usize, step: usize, chunks: []const u32, marks: []const u32) usize {
    var n: usize = 0;
    var at = from;
    while (at < len) {
        const end = pass.chunkEnd(at, len, step, chunks, marks);
        std.debug.assert(end > at);
        out[n] = end;
        n += 1;
        at = end;
    }
    return n;
}

test "nemotron: a fresh pass cuts at every message start, not at the grid step" {
    // A 4096 grid would run all 1000 rows in one chunk; the starts split it.
    const chunks = [_]u32{ 128, 300, 640 };
    var out: [8]usize = undefined;
    const n = walk(&out, 0, 1000, 4096, &chunks, &.{});
    try t.expectEqualSlices(usize, &.{ 128, 300, 640, 1000 }, out[0..n]);
}

test "nemotron: a resumed pass from a kept start cuts at the next start (the resume point)" {
    // A resume at 300 must still visit the later message start at 640.
    const chunks = [_]u32{ 128, 300, 640 };
    const marks = [_]u32{640};
    var out: [8]usize = undefined;
    const n = walk(&out, 300, 1000, 4096, &chunks, &marks);
    try t.expectEqualSlices(usize, &.{ 640, 1000 }, out[0..n]);
    // every cut before the prompt's end is a message start, so a state is kept there
    for (out[0 .. n - 1]) |end| try t.expect(std.mem.indexOfScalar(u32, &chunks, @intCast(end)) != null);
}

test "nemotron: a store mark off the grid ends a chunk on resume" {
    const marks = [_]u32{ 1000, 2200 };
    var out: [8]usize = undefined;
    const n = walk(&out, 0, 5000, 2048, &.{}, &marks);
    try t.expectEqualSlices(usize, &.{ 1000, 2200, 4248, 5000 }, out[0..n]);
}

test "nemotron: with no message starts the grid step is unchanged" {
    var out: [8]usize = undefined;
    const n = walk(&out, 0, 5000, 2048, &.{}, &.{});
    try t.expectEqualSlices(usize, &.{ 2048, 4096, 5000 }, out[0..n]);
}

test "nemotron: cache-off passes still cut at the request's starts through the shared source" {
    // No keep callback and no store marks: only the stream's chunks can end a chunk.
    const prompt = [_]u32{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    const starts = [_]u32{ 1, 2, 3 };
    var s = try lanes.Stream.init(t.allocator, .{
        .id = "cache-off", .prompt = &prompt, .max_new = 1, .drafts = false, .chunks = &starts,
    });
    defer s.deinit(t.allocator);
    const points = lanes.cutPoints(&s);
    try t.expectEqualSlices(u32, &starts, points);
    var out: [8]usize = undefined;
    const n = walk(&out, 0, prompt.len, state.pass_rows, points, &.{});
    try t.expectEqualSlices(usize, &.{ 1, 2, 3, 12 }, out[0..n]);
}

// -- native-path host control -----------------------------------------------------------------------------
// The helpers above are the same ones serialChunks runs. These tests drive them with the family's real
// unplanned step and mark lookahead, an opaque Saver callback (never a cast Stream), and a cut-invariant
// row fold, so a fresh cache-off pass, a resume at a message start and the uncut main path are compared.

const native = @import("cuda_native.zig");
const marks_mod = @import("cuda_marks.zig");
const engine = @import("cuda_engine.zig");

/// The resume_test Saver shape: the keep context is opaque, as a standalone caller passes it.
const Saver = struct {
    at: [64]u32 = undefined,
    n: usize = 0,

    fn call(ctx: *anyopaque, row: u32) void {
        const self: *Saver = @ptrCast(@alignCast(ctx)); // a Saver, never a lanes.Stream
        self.at[self.n] = row;
        self.n += 1;
    }
};

/// A row's bits depend on its token and absolute position, never on where a pass cut (cut-invariance).
fn fold(h: u64, row: usize, tok: u32) u64 {
    return h *% 6364136223846793005 +% @as(u64, tok) *% 0x9e3779b97f4a7c15 +% @as(u64, row) +% 1;
}
fn prefix(tokens: []const u32, end: usize) u64 {
    var h: u64 = 0x243f6a8885a308d3;
    for (tokens[0..end], 0..) |tok, row| h = fold(h, row, tok);
    return h;
}
fn fillTokens(comptime len: usize) [len]u32 {
    var out: [len]u32 = undefined;
    for (&out, 0..) |*v, i| v.* = @intCast(i % 977 + 1);
    return out;
}

/// Host mirror of serialChunks' plan: real chunkEnd, real lookahead, real keep callback order.
const Plan = struct {
    ends: [64]usize = undefined,
    nends: usize = 0,
    saver: Saver = .{},
    digest: u64 = 0,
};
fn plan(from: usize, len: usize, tokens: []const u32, chunks: []const u32, store_marks: []const u32, cut: bool) Plan {
    var p: Plan = .{};
    const keep: ?engine.Keep = if (store_marks.len == 0) null else .{ .at = store_marks, .ctx = &p.saver, .call = Saver.call, .cut = cut };
    var h = if (from == 0) @as(u64, 0x243f6a8885a308d3) else prefix(tokens, from); // a restored Saver state
    var s = from;
    while (s < len) {
        // Unplanned invariant family: its step is prompt_rows, shortened by real mark lookahead.
        const step: usize = if (pass.inside(keep, s, s + state.pass_rows) < marks_mod.max) state.pass_rows else state.prefill_rows;
        const forced: []const u32 = if (keep) |k| (if (k.cut) k.at else &.{}) else &.{};
        const end = pass.chunkEnd(s, len, step, chunks, forced);
        std.debug.assert(end > s);
        p.ends[p.nends] = end;
        p.nends += 1;
        if (keep) |k| {
            for (k.at) |m| if (m > s and m < end) k.call(k.ctx, m); // arm: interior captures first
        }
        for (s..end) |row| h = fold(h, row, tokens[row]);
        if (keep) |k| {
            if (std.mem.indexOfScalar(u32, k.at, @intCast(end)) != null or std.mem.indexOfScalar(u32, chunks, @intCast(end)) != null) k.call(k.ctx, @intCast(end));
        }
        s = end;
    }
    p.digest = h;
    return p;
}

test "request starts: native cold with message starts equals the uncut main path" {
    // The real family is unplanned and cut-invariant: its request step is prompt_rows, with no snap_grid.
    try t.expect(native.snap_cut_invariant);
    try t.expect(!@hasDecl(native, "snap_grid"));
    try t.expectEqual(@as(u32, state.pass_rows), native.prompt_rows);
    const len = 9000;
    const toks = fillTokens(len);
    const starts = [_]u32{ 700, 1024, 2100, 4096, 5700, 8192 };
    const main = plan(0, len, &toks, &.{}, &.{}, false); // main: no starts reach the pass
    const cold = plan(0, len, &toks, &starts, &.{}, false); // candidate: cache-off, starts carried explicitly
    try t.expectEqual(main.digest, cold.digest); // cold == main: the same rows in the same order
    for (starts) |st| try t.expect(std.mem.indexOfScalar(usize, cold.ends[0..cold.nends], @as(usize, st)) != null);
    // Main control: the off-grid starts are not ends when the pass is not handed the list.
    for (starts) |st| if (st % native.prompt_rows != 0) try t.expect(std.mem.indexOfScalar(usize, main.ends[0..main.nends], @as(usize, st)) == null);
}

test "request starts: a resume at a kept message start equals a fresh cache-off pass" {
    const len = 9000;
    const toks = fillTokens(len);
    const starts = [_]u32{ 700, 2100, 4096, 5700, 8192 };
    const marks = [_]u32{ 2100, 5700 }; // store marks at two message starts, as a kept request has
    const fresh = plan(0, len, &toks, &starts, &marks, false);
    try t.expect(std.mem.indexOfScalar(u32, fresh.saver.at[0..fresh.saver.n], 2100) != null);
    const resumed = plan(2100, len, &toks, &starts, &marks, false); // restore the Saver state at 2100
    try t.expectEqual(fresh.digest, resumed.digest); // resumed == fresh cache-off
    for (starts) |st| if (st > 2100) try t.expect(std.mem.indexOfScalar(u32, resumed.saver.at[0..resumed.saver.n], st) != null);
}

test "request starts: cut=false captures marks inside, cut=true ends chunks, both fold the same rows" {
    const len = 9000;
    const toks = fillTokens(len);
    const store_marks = [_]u32{ 700, 2100, 3300 }; // inside the first grid pass, off any grid line
    const capture = plan(0, len, &toks, &.{}, &store_marks, false);
    try t.expectEqual(@as(usize, state.pass_rows), capture.ends[0]); // no mark ended the chunk
    for (store_marks) |m| try t.expect(std.mem.indexOfScalar(u32, capture.saver.at[0..capture.saver.n], m) != null);
    const forced = plan(0, len, &toks, &.{}, &store_marks, true);
    try t.expectEqualSlices(usize, &.{ 700, 2100, 3300, 7396, 9000 }, forced.ends[0..forced.nends]);
    try t.expectEqual(capture.digest, forced.digest); // the device control compares distinct captured states, not rows
}

test "request starts: main control — grid-only planning cannot keep a named message start" {
    // Holds only when the pass is handed the starts. On pristine main prefill has no `chunks` argument
    // and no way to cut at 700 or 2100, so this control fails there.
    const len = 5000;
    const toks = fillTokens(len);
    const starts = [_]u32{ 700, 2100 };
    const c = plan(0, len, &toks, &starts, &.{}, false);
    try t.expectEqualSlices(usize, &.{ 700, 2100, 4096, 5000 }, c.ends[0..c.nends]);
    const m = plan(0, len, &toks, &.{}, &.{}, false);
    try t.expectEqualSlices(usize, &.{ 4096, 5000 }, m.ends[0..m.nends]);
}
