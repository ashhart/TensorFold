//! One per-request cache report, with pressure changes and reminders rated apart from its ordinary counters.
const std = @import("std");
const pc = @import("prompt_cache.zig");

pub const Level = enum { info, warning };
pub const Line = struct { level: Level, text: []const u8 };
pub const State = struct {
    holding: bool = false,
    warned_at: ?i96 = null,
};
pub const reminder_ns: i96 = 60 * std.time.ns_per_s;

/// One log line after a prompt pass: where it resumed, how many states it kept, and what the store holds.
pub fn report(s: *pc.Store, prompt: usize, from: u32, kept: u64, now: i96) void {
    if (@import("builtin").is_test) return;
    var buf: [1024]u8 = undefined;
    const line = renderAt(s, prompt, from, kept, now, &buf) catch return;
    switch (line.level) {
        .info => std.log.info("{s}", .{line.text}),
        .warning => std.log.warn("{s}", .{line.text}),
    }
}

/// The report at a caller's timestamp; tests inject time without waiting or writing to stderr.
pub fn renderAt(s: *pc.Store, prompt: usize, from: u32, kept: u64, now: i96, buf: []u8) !Line {
    const state = &s.report_state;
    if (!s.hold and state.holding) {
        state.* = .{};
        return .{ .level = .info, .text = "prompt cache: memory pressure cleared; new prompt states are kept again" };
    }
    const due = if (state.warned_at) |at| now - at >= reminder_ns else true;
    if (s.hold and due) {
        const reminder = state.holding;
        state.holding = true;
        state.warned_at = now;
        if (reminder) return .{ .level = .warning, .text = "prompt cache: memory pressure still prevents new prompt states" };
        return .{ .level = .warning, .text = try std.fmt.bufPrint(buf, summary ++ "; memory pressure: new prompt states are not kept", .{ prompt, from, kept, s.entries.items.len, s.held >> 20, s.spare() >> 20, s.budget >> 20, s.counts.hits, s.counts.misses, s.counts.evicted, s.counts.refused, s.counts.failed, s.counts.trimmed }) };
    }
    state.holding = s.hold;
    return .{ .level = .info, .text = try std.fmt.bufPrint(buf, summary, .{ prompt, from, kept, s.entries.items.len, s.held >> 20, s.spare() >> 20, s.budget >> 20, s.counts.hits, s.counts.misses, s.counts.evicted, s.counts.refused, s.counts.failed, s.counts.trimmed }) };
}

const summary = "prompt cache: {d} tokens, resumed at {d}, kept {d}; {d} states, {d} MiB and {d} MiB spare of {d} MiB (hits {d}, misses {d}, evicted {d}, refused {d}, failed {d}, trimmed {d})";
