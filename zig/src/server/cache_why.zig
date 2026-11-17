//! A reply's prompt-cache reason (#3686): the TF_CACHE_WHY object, the done line's fields and the /metrics series.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const Server = @import("server.zig").Server;
const Why = api.prompt_cache.why.Why;
const Reason = api.prompt_cache.why.Reason;
const Allocator = std.mem.Allocator;

/// The `tensorfold.prompt_cache` object: the reason, N, what resumed and was lost, and the message holding N.
pub fn value(srv: *const Server, a: Allocator, w: Why, prompt: []const u32) !json.Value {
    const o = try json.newObject(a);
    try o.put(a, "reason", .{ .string = @tagName(w.reason) });
    try o.put(a, "shared", try json.intValue(a, w.shared));
    try o.put(a, "resumed", try json.intValue(a, w.resumed));
    try o.put(a, "lost", try json.intValue(a, w.lost()));
    if (messageAt(srv.chunks.openers, prompt, w.shared)) |m| try o.put(a, "message", try json.intValue(a, m));
    return .{ .object = o };
}

/// The rendered message holding token `n`, counting the template's message openers (null: openers unknown).
pub fn messageAt(openers: []const u32, prompt: []const u32, n: u32) ?u32 {
    if (openers.len == 0 or prompt.len == 0) return null;
    var seen: u32 = 0;
    for (prompt[0..@min(@as(usize, n) + 1, prompt.len)]) |t| seen += @intFromBool(std.mem.indexOfScalar(u32, openers, t) != null);
    return if (seen == 0) null else seen - 1;
}

/// The done line's ` cache=<reason>`, with ` at=<N> lost=<L>` past `none`: positions only, never token ids.
pub fn logged(buf: []u8, why: ?Why) []const u8 {
    const w = why orelse return "";
    if (w.reason == .none) return std.fmt.bufPrint(buf, " cache=none", .{}) catch "";
    return std.fmt.bufPrint(buf, " cache={s} at={d} lost={d}", .{ @tagName(w.reason), w.shared, w.lost() }) catch "";
}

/// Requests and lost tokens by reason, for /metrics: one counter a reason, so the labels are bounded.
pub const Counts = struct {
    lookups: [api.prompt_cache.why.reasons]u64 = @splat(0),
    lost: [api.prompt_cache.why.reasons]u64 = @splat(0),

    pub fn note(c: *Counts, w: Why) void {
        c.lookups[@backingInt(w.reason)] += 1;
        c.lost[@backingInt(w.reason)] += w.lost();
    }

    pub fn render(c: *const Counts, w: *std.Io.Writer, prefix: []const u8) !void {
        try w.print("# HELP {s}prompt_cache_lookups_total Finished requests by why their prompt resumed no further.\n", .{prefix});
        try w.print("# TYPE {s}prompt_cache_lookups_total counter\n", .{prefix});
        for (std.enums.values(Reason)) |r| try w.print("{s}prompt_cache_lookups_total{{reason=\"{s}\"}} {d}\n", .{ prefix, @tagName(r), c.lookups[@backingInt(r)] });
        try w.print("# HELP {s}prompt_cache_lost_tokens_total Prompt tokens shared with a kept prompt but prefilled again, by reason.\n", .{prefix});
        try w.print("# TYPE {s}prompt_cache_lost_tokens_total counter\n", .{prefix});
        for (std.enums.values(Reason)) |r| try w.print("{s}prompt_cache_lost_tokens_total{{reason=\"{s}\"}} {d}\n", .{ prefix, @tagName(r), c.lost[@backingInt(r)] });
    }
};
