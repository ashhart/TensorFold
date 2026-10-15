//! Serial greedy decode over any family's engine on the Intel GPU: prompt row by row, then a token a step or forced.

const std = @import("std");
const stop = @import("stop.zig");

pub const Options = struct {
    stop_eos: bool = true,
    /// Teacher forcing: feed these tokens after the prompt instead of the model's own; the end token never stops.
    force: ?[]const u32 = null,
    /// Keep each step's five largest logits.
    top5: bool = false,
    /// Engines with `prefillWindows` run the prompt head in windows of this many rows (`--prefill`).
    prefill_rows: u32 = 0,
};

pub const Top = struct { id: u32, v: f32 };

pub const Result = struct {
    tokens: []u32,
    /// Per produced token, when asked for.
    top5: [][5]Top,
    /// Forced runs with top5: per step the log-probability of the forced token and of our top-1 (log-softmax, fp32).
    lpf: [][2]f64 = &.{},
    prefill_seconds: f64,
    decode_seconds: f64,
    rounds: usize,
    /// A stop was asked for (SIGINT, SIGTERM or the stop file): the tokens so far, every launch finished.
    interrupted: bool = false,
};

pub fn seconds(io: std.Io, since: std.Io.Timestamp) f64 {
    const now = std.Io.Clock.awake.now(io);
    return @as(f64, @floatFromInt(now.toNanoseconds() - since.toNanoseconds())) / 1e9;
}

/// The five largest logits, best first; the lowest index wins a tie.
pub fn top5(logits: []const f32) [5]Top {
    var best: [5]Top = @splat(.{ .id = 0, .v = -std.math.inf(f32) });
    for (logits, 0..) |v, i| {
        if (v <= best[4].v) continue;
        var j: usize = 4;
        while (j > 0 and v > best[j - 1].v) : (j -= 1) best[j] = best[j - 1];
        best[j] = .{ .id = @intCast(i), .v = v };
    }
    return best;
}

/// `count` tokens (the first from the prompt's last row); forced runs give one token per forced id.
pub fn generate(gpa: std.mem.Allocator, io: std.Io, e: anytype, prompt: []const u32, count: usize, o: Options) !Result {
    if (prompt.len == 0) return error.NoPromptTokens;
    const n = if (o.force) |f| f.len else count;
    if (n == 0) return error.NothingToGenerate;
    if (prompt.len + n - 1 > e.max_len) return error.ContextTooSmall;
    const stop_eos = o.stop_eos and o.force == null;
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var tops: std.ArrayList([5]Top) = .empty;
    errdefer tops.deinit(gpa);
    var lpfs: std.ArrayList([2]f64) = .empty;
    errdefer lpfs.deinit(gpa);
    const logits = try gpa.alloc(f32, if (o.top5) e.vocab() else 0);
    defer gpa.free(logits);
    try e.reset();
    const t0 = std.Io.Clock.awake.now(io);
    // the token's host copy is reused by the next feed, so each prompt row is waited for
    const windowed = comptime @hasDecl(@TypeOf(e.*), "prefillWindows");
    if (windowed and o.prefill_rows > 0 and prompt.len > 1) try e.prefillWindows(prompt[0 .. prompt.len - 1], o.prefill_rows) else for (prompt[0 .. prompt.len - 1]) |t| {
        if (stop.requested()) return stopped(gpa, &out, &tops, io, t0);
        try e.feed(t, false);
        try e.sync();
    }
    var next = try step(e, prompt[prompt.len - 1]);
    try out.append(gpa, next);
    if (o.top5) try record(gpa, e, logits, o, &tops, &lpfs, out.items.len - 1);
    const prefill_s = seconds(io, t0);
    const t1 = std.Io.Clock.awake.now(io);
    while (out.items.len < n and !(stop_eos and e.isEos(next))) {
        if (stop.requested()) break;
        const fed = if (o.force) |f| f[out.items.len - 1] else next;
        next = try step(e, fed);
        try out.append(gpa, next);
        if (o.top5) try record(gpa, e, logits, o, &tops, &lpfs, out.items.len - 1);
    }
    const rounds = out.items.len;
    return .{ .tokens = try out.toOwnedSlice(gpa), .top5 = try tops.toOwnedSlice(gpa), .lpf = try lpfs.toOwnedSlice(gpa), .prefill_seconds = prefill_s, .decode_seconds = seconds(io, t1), .rounds = rounds, .interrupted = stop.requested() and out.items.len < n };
}

/// The result of a run stopped inside the prompt: no tokens.
fn stopped(gpa: std.mem.Allocator, out: *std.ArrayList(u32), tops: *std.ArrayList([5]Top), io: std.Io, t0: std.Io.Timestamp) !Result {
    return .{ .tokens = try out.toOwnedSlice(gpa), .top5 = try tops.toOwnedSlice(gpa), .prefill_seconds = seconds(io, t0), .decode_seconds = 0, .rounds = 0, .interrupted = true };
}

fn step(e: anytype, token: u32) !u32 {
    try e.feed(token, true);
    return e.argmax();
}

fn record(gpa: std.mem.Allocator, e: anytype, logits: []f32, o: Options, tops: *std.ArrayList([5]Top), lpfs: *std.ArrayList([2]f64), idx: usize) !void {
    try e.fetchLogits(logits);
    const t = top5(logits);
    try tops.append(gpa, t);
    if (o.force) |f| if (idx < f.len) {
        const mx: f64 = t[0].v;
        var se: f64 = 0;
        for (logits) |v| se += @exp(@as(f64, v) - mx);
        const lse = mx + @log(se);
        try lpfs.append(gpa, .{ @as(f64, logits[f[idx]]) - lse, @as(f64, t[0].v) - lse });
    };
}

test "top5 keeps the lowest index among equal logits" {
    const l = [_]f32{ 1, 3, 3, 2, 3, 0, 2.5 };
    const t = top5(&l);
    try std.testing.expectEqual(@as(u32, 1), t[0].id);
    try std.testing.expectEqual(@as(u32, 2), t[1].id);
    try std.testing.expectEqual(@as(u32, 4), t[2].id);
    try std.testing.expectEqual(@as(u32, 6), t[3].id);
    try std.testing.expectEqual(@as(u32, 3), t[4].id);
}
