//! A lane host's status counts: running and waiting requests, the 2 s decode and prompt rates, each stream's tokens.
const std = @import("std");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;

/// Tokens a round landed, and when.
pub const Mark = struct { at: i96, tokens: u64 };
const window_ns: i96 = 2 * std.time.ns_per_s;

/// Engine.status: the counts, read under the host's lock.
pub fn read(ctx: *anyopaque, out: *api.Status, stream_tokens: []u32) void {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    h.mutex.lockUncancelable(h.io);
    defer h.mutex.unlock(h.io);
    const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
    var tokens: u64 = 0;
    for (h.decoded.items) |m| {
        if (m.at >= now - window_ns) tokens += m.tokens;
    }
    const n = @min(stream_tokens.len, h.live_tokens.items.len);
    @memcpy(stream_tokens[0..n], h.live_tokens.items[0..n]);
    out.* = .{
        .running = @intCast(h.admitted.items.len),
        .waiting = @intCast(h.queued.items.len),
        .decode_tokens_per_second = @as(f64, @floatFromInt(tokens)) / 2.0,
        .prefill_tokens_per_second = if (now - h.prefill_at <= window_ns) h.prefill_rate else 0,
        .preemptions = 0,
        .streams = n,
        .generation_tokens = h.live_generated,
    };
}

/// `n` tokens delivered now; marks older than the window drop.
pub fn noteDecoded(h: *LaneHost, n: usize) void {
    const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
    h.mutex.lockUncancelable(h.io);
    defer h.mutex.unlock(h.io);
    var keep: usize = 0;
    for (h.decoded.items) |m| {
        if (m.at < now - window_ns) continue;
        h.decoded.items[keep] = m;
        keep += 1;
    }
    h.decoded.shrinkRetainingCapacity(keep);
    h.decoded.append(h.gpa, .{ .at = now, .tokens = n }) catch {};
}

/// Each started stream's context and generated tokens, for the next status read.
pub fn noteLive(h: *LaneHost) void {
    h.mutex.lockUncancelable(h.io);
    defer h.mutex.unlock(h.io);
    h.live_tokens.clearRetainingCapacity();
    h.live_generated = 0;
    for (h.admitted.items) |job| if (job.started) {
        h.live_tokens.append(h.gpa, @intCast(job.stream.context.items.len)) catch {};
        h.live_generated += @intCast(job.stream.emitted().len);
    };
}
