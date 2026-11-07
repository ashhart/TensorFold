//! A drafter's taps: after a prompt and a round, a stream's ring holds the tapped layers' residual rows, bit for bit.

const std = @import("std");
const hip = @import("hip");
const qwen35 = @import("qwen3_5");

const Engine = qwen35.engine.Engine;

/// DFlash2's taps on a 64-layer target; elsewhere the first, middle and last layers.
pub fn layersOf(n_layers: usize, out: *[5]u32) []const u32 {
    if (n_layers == 64) {
        out.* = .{ 5, 19, 33, 47, 61 };
        return out;
    }
    out[0..3].* = .{ 0, @intCast(n_layers / 2), @intCast(n_layers - 1) };
    return out[0..3];
}

/// The tapped layers' residual rows of one forward, (taps, rows, hidden) bytes, as the trace saw them.
const Sink = struct {
    e: *Engine,
    taps: []const u32,
    bytes: []u8,
    rows: usize,

    fn width(s: *const Sink) usize {
        return s.e.model().spec.hidden * 2;
    }

    fn layer(ctx: *anyopaque, index: usize, x: hip.ops.Tensor, _: usize) anyerror!void {
        const s: *Sink = @ptrCast(@alignCast(ctx));
        const ti = std.mem.indexOfScalar(u32, s.taps, @intCast(index)) orelse return;
        try s.e.stream.synchronize();
        const n = s.rows * s.width();
        try hip.runtime.check(s.e.driver.api.hipMemcpyDtoH(s.bytes[ti * n ..].ptr, @ptrFromInt(x.ptr), n));
    }
};

/// Where the ring and the trace first differ.
pub const Diff = struct { tap: usize, position: usize, column: usize };

/// The ring's rows at positions `from ..` against the sink's rows; null when every byte matches.
fn compare(e: *Engine, ring: []const u8, sink: *const Sink, from: usize) ?Diff {
    const t = &e.model().taps;
    const w = sink.width();
    for (0..sink.taps.len) |ti| for (0..sink.rows) |r| {
        const p = from + r;
        const got = ring[((p % t.rows) * t.n + ti) * w ..][0..w];
        const want = sink.bytes[(ti * sink.rows + r) * w ..][0..w];
        if (std.mem.indexOfDiff(u8, got, want)) |col| return .{ .tap = ti, .position = p, .column = col / 2 };
    };
    return null;
}

/// A prompt of `ids` but its last `n` tokens, then a round of those `n`: the ring against both forwards' residuals.
pub fn runCase(gpa: std.mem.Allocator, e: *Engine, ids: []const u32, n: usize) !?Diff {
    if (n < 1 or n > 16 or ids.len <= n) return error.PromptTooShort;
    // the trace waits on the stream layer by layer, which a capture cannot
    const was = e.o.graphs;
    e.o.graphs = false;
    defer e.o.graphs = was;
    var buf: [5]u32 = undefined;
    const taps = layersOf(e.model().spec.n_layers, &buf);
    try e.setTaps(taps);
    defer e.setTaps(&.{}) catch {};
    var caches = try e.newCaches(ids.len + 8);
    defer caches.deinit(gpa);
    const prompt = ids[0 .. ids.len - n];
    const width = e.model().spec.hidden * 2;

    // the prompt pass, traced
    var sink: Sink = .{ .e = e, .taps = taps, .bytes = try gpa.alloc(u8, taps.len * prompt.len * width), .rows = prompt.len };
    defer gpa.free(sink.bytes);
    try promptPass(e, &caches, prompt, .{ .ctx = &sink, .layer = Sink.layer });
    const ring = try gpa.alloc(u8, caches.tap.?.len);
    defer gpa.free(ring);
    try caches.tap.?.download(0, ring);
    if (compare(e, ring, &sink, 0)) |d| return d;

    // one round of the last n tokens, traced
    var round: Sink = .{ .e = e, .taps = taps, .bytes = try gpa.alloc(u8, taps.len * n * width), .rows = n };
    defer gpa.free(round.bytes);
    e.round.trace = .{ .ctx = &round, .layer = Sink.layer };
    defer e.round.trace = null;
    try roundPass(e, &caches, prompt.len, ids[prompt.len..]);
    try caches.tap.?.download(0, ring);
    return compare(e, ring, &round, prompt.len);
}

/// A second stream's round, replayed from the graph of the first's: its ring against both rounds run eagerly.
pub fn graphsCase(gpa: std.mem.Allocator, e: *Engine, ids: []const u32, n: usize) !?Diff {
    if (n < 1 or n > 16 or ids.len <= n) return error.PromptTooShort;
    var buf: [5]u32 = undefined;
    const taps = layersOf(e.model().spec.n_layers, &buf);
    const graph = try secondRing(gpa, e, taps, ids, n, true);
    defer gpa.free(graph);
    const eager = try secondRing(gpa, e, taps, ids, n, false);
    defer gpa.free(eager);
    const at = std.mem.indexOfDiff(u8, graph, eager) orelse return null;
    // a ring row holds every tap's row; these positions stay below the ring's rows
    const w = e.model().spec.hidden * 2;
    return .{ .tap = at / w % taps.len, .position = at / w / taps.len, .column = at % w / 2 };
}

/// A prompt and a round on two streams, graphs on (the first captures, the second replays) or off: the second's ring.
fn secondRing(gpa: std.mem.Allocator, e: *Engine, taps: []const u32, ids: []const u32, n: usize, graphs: bool) ![]u8 {
    const was = e.o.graphs;
    e.o.graphs = graphs;
    defer e.o.graphs = was;
    try e.setTaps(taps);
    defer e.setTaps(&.{}) catch {};
    // both streams live at once, so the second's ring is never the first's memory again
    var first = try e.newCaches(ids.len + 8);
    defer first.deinit(gpa);
    var second = try e.newCaches(ids.len + 8);
    defer second.deinit(gpa);
    const at = ids.len - n;
    for ([_]*qwen35.state.Caches{ &first, &second }) |c| {
        try promptPass(e, c, ids[0..at], null);
        try roundPass(e, c, at, ids[at..]);
    }
    if (graphs and e.graphs.replayed != 1) return error.RoundNotReplayed;
    const ring = try gpa.alloc(u8, second.tap.?.len);
    errdefer gpa.free(ring);
    try second.tap.?.download(0, ring);
    return ring;
}

/// The prompt pass of fresh caches: Engine.advance with `trace` in place of the cancel poll.
fn promptPass(e: *Engine, caches: *qwen35.state.Caches, prompt: []const u32, trace: ?qwen35.forward.Trace) !void {
    e.prompts.reset();
    @memcpy(e.ids.slice(u32)[0..prompt.len], prompt);
    try e.ids_dev.uploadAsync(0, e.ids.view(0, 4 * prompt.len), e.stream);
    _ = try qwen35.forward.span(e.ops(&e.prompts), e.model(), caches, e.ids_dev.base(), prompt.len, 0, trace);
    try e.stream.synchronize();
}

/// One round of `tokens` from `pos`, every row kept.
fn roundPass(e: *Engine, caches: *qwen35.state.Caches, pos: usize, tokens: []const u32) !void {
    var reqs: [16]qwen35.draw.Request = undefined;
    var out: [16]u32 = undefined;
    for (0..tokens.len) |i| reqs[i] = .{ .sampling = null, .position = pos + i + 1 };
    const rows = [_]Engine.Rows{.{ .caches = caches, .pos = pos, .tokens = tokens }};
    _ = try e.verify(&rows, reqs[0..tokens.len], out[0..tokens.len]);
    try e.stream.synchronize();
    e.keep(0, tokens.len);
    try e.flush();
}
