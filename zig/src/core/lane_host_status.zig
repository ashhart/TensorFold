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
        .healthy = !h.closing and !h.stale and (if (h.health) |source| source.read(source.ctx) else true),
        .running = @intCast(h.admitted.items.len),
        .waiting = @intCast(h.queued.items.len),
        .decode_tokens_per_second = @as(f64, @floatFromInt(tokens)) / 2.0,
        .prefill_tokens_per_second = if (now - h.prefill_at <= window_ns) h.prefill_rate else 0,
        .preemptions = 0,
        .streams = n,
        .generation_tokens = h.live_generated,
    };
    fillTotals(h, out);
}

/// A job whose prompt pass has not reported yet is filling; the engine thread publishes that with `fill_on`.
fn fillTotals(h: *LaneHost, out: *api.Status) void {
    var total: u64 = 0;
    var left: u64 = 0;
    var filling = false;
    var run: u64 = 0;
    const reported = h.core.backend.vtable.reports_fill; // a backend that never stores `fill_at` has no honest totals
    for (h.admitted.items) |job| {
        if (!job.fill_on.load(.acquire) or job.request.background) continue; // a warm pass prefills the next turn: no fill
        filling = true;
        var fields: [16]u8 = undefined; // id and seq hashed apart: XOR of the raw values collides (2^0 == 3^1)
        std.mem.writeInt(u64, fields[0..8], job.id, .little);
        std.mem.writeInt(u64, fields[8..16], job.seq, .little);
        run = (run *% 0x9E3779B97F4A7C15) ^ std.hash.Wyhash.hash(0, &fields);
        if (!reported) continue; // no counter for it: its prompt stays out of the totals rather than guessed
        const prompt = job.request.prompt.len;
        total += prompt;
        left += prompt -| job.stream.fill_at.load(.monotonic);
    }
    if (!filling) return;
    if (reported) {
        out.fill_prompt_tokens = @intCast(total);
        out.fill_tokens_left = @intCast(left);
    } else out.fill_unknown = true;
    run &= (1 << 53) - 1; // JSON numbers stay exact to 2^53: a wider marker loses bits in the page's JavaScript
    out.fill_run = if (run == 0) 1 else run;
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
