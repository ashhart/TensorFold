//! The Spark stats wire: the Spark server's ``tensorfold`` keys, beside a reply's own for DeepSeek-V4 checkpoints.
const std = @import("std");
const json = @import("json");
const chat = @import("chat.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// Whether replies carry the Spark keys: TENSORFOLD_WIRE spark or native (tensorfold) wins, else the family decides.
pub fn enabled(model_type: []const u8, setting: ?[]const u8) ?bool {
    const t = std.mem.trim(u8, setting orelse "", " \t");
    if (t.len == 0) return std.mem.eql(u8, model_type, "deepseek_v41") or std.mem.eql(u8, model_type, "deepseek_v4");
    if (std.ascii.eqlIgnoreCase(t, "spark")) return true;
    if (std.ascii.eqlIgnoreCase(t, "native") or std.ascii.eqlIgnoreCase(t, "tensorfold")) return false;
    return null; // neither: the server refuses to start
}

/// The Spark server's prompt, cached, ttft_s, finish, completion, prefill_s, decode_s and draft counters, in its order.
pub fn add(a: Allocator, reply: *const chat.Reply) Allocator.Error!void {
    if (reply.runtime != .object) return;
    const block = reply.runtime;
    const o = block.object;
    try o.put(a, "prompt", try json.intValue(a, reply.prompt_tokens));
    try o.put(a, "cached", try json.intValue(a, reply.cached_tokens));
    if (secondsOf(block, "time_to_first_token")) |t| try o.put(a, "ttft_s", .{ .float = round4(t) });
    try o.put(a, "finish", .{ .string = if (std.mem.eql(u8, reply.finish_reason, "cancelled")) "cancelled" else "done" });
    try o.put(a, "completion", try json.intValue(a, reply.completion_tokens));
    if (secondsOf(block, "prefill_seconds")) |p| try o.put(a, "prefill_s", .{ .float = round4(p) });
    if (secondsOf(block, "seconds")) |d| try o.put(a, "decode_s", .{ .float = round4(d) }); // from the prefill's end
    for ([_][]const u8{ "rounds", "drafted", "accepted" }) |k| try o.put(a, k, try json.intValue(a, count(reply.speculative, k)));
}

fn secondsOf(block: Value, key: []const u8) ?f64 {
    const v = block.field(key) orelse return null;
    return v.float64();
}

fn count(spec: Value, key: []const u8) i64 {
    const v = spec.get(key) orelse return 0;
    return v.int64() orelse 0;
}

/// Python's ``round(x, 4)``, as the Spark server rounds its seconds.
fn round4(x: f64) f64 {
    var buf: [64]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, "{d:.4}", .{x}) catch return x;
    return std.fmt.parseFloat(f64, text) catch x;
}

test "TENSORFOLD_WIRE overrides the family's default" {
    try std.testing.expectEqual(@as(?bool, true), enabled("deepseek_v41", null));
    try std.testing.expectEqual(@as(?bool, true), enabled("deepseek_v4", ""));
    try std.testing.expectEqual(@as(?bool, false), enabled("qwen3", null));
    try std.testing.expectEqual(@as(?bool, true), enabled("qwen3", "spark"));
    try std.testing.expectEqual(@as(?bool, false), enabled("deepseek_v41", "native"));
    try std.testing.expectEqual(@as(?bool, false), enabled("deepseek_v41", " tensorfold "));
    try std.testing.expectEqual(@as(?bool, null), enabled("deepseek_v41", "sparks"));
}
