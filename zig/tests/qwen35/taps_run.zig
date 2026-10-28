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

    // the prompt pass, traced: Engine.advance with a trace in place of the cancel poll
    var sink: Sink = .{ .e = e, .taps = taps, .bytes = try gpa.alloc(u8, taps.len * prompt.len * width), .rows = prompt.len };
    defer gpa.free(sink.bytes);
    e.prompts.reset();
    @memcpy(e.ids.slice(u32)[0..prompt.len], prompt);
    try e.ids_dev.uploadAsync(0, e.ids.view(0, 4 * prompt.len), e.stream);
    _ = try qwen35.forward.span(e.ops(&e.prompts), e.model(), &caches, e.ids_dev.base(), prompt.len, 0, .{ .ctx = &sink, .layer = Sink.layer });
    try e.stream.synchronize();
    const ring = try gpa.alloc(u8, caches.tap.?.len);
    defer gpa.free(ring);
    try caches.tap.?.download(0, ring);
    if (compare(e, ring, &sink, 0)) |d| return d;

    // one round of the last n tokens, traced
    var round: Sink = .{ .e = e, .taps = taps, .bytes = try gpa.alloc(u8, taps.len * n * width), .rows = n };
    defer gpa.free(round.bytes);
    var reqs: [16]qwen35.draw.Request = undefined;
    var out: [16]u32 = undefined;
    for (0..n) |i| reqs[i] = .{ .sampling = null, .position = prompt.len + i + 1 };
    const rows = [_]Engine.Rows{.{ .caches = &caches, .pos = prompt.len, .tokens = ids[prompt.len..] }};
    e.round.trace = .{ .ctx = &round, .layer = Sink.layer };
    defer e.round.trace = null;
    _ = try e.verify(&rows, reqs[0..n], out[0..n]);
    try e.stream.synchronize();
    try caches.tap.?.download(0, ring);
    e.keep(0, n);
    try e.flush();
    return compare(e, ring, &round, prompt.len);
}
