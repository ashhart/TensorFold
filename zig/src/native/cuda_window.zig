//! A CUDA family's context window: the model's own (config.json), and what --context makes of it.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// The model's window (config.json's max_position_embeddings, text_config's first), 0 when it names none.
pub fn modelContext(a: Allocator, io: std.Io, dir: []const u8) i64 {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return 0;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return 0;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return 0;
    if (doc != .object) return 0;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    const limit = text.object.get("max_position_embeddings") orelse doc.object.get("max_position_embeddings") orelse return 0;
    return if (limit == .integer and limit.integer > 0) limit.integer else 0;
}

/// The window a family's own row raises --context's ceiling to, and the switch its refusal names; none without one.
pub const Ceiling = struct { window: ?u64 = null, switch_name: []const u8 = "" };

/// A family's raised ceiling, read at its open (GLM's TF_GLM_CONTEXT_CEILING); a malformed row is the family's refusal.
pub fn familyCeiling(comptime F: type, problem: *[]const u8) error{Refused}!Ceiling {
    if (comptime !@hasDecl(F, "raisedCeiling")) return .{};
    const window = F.raisedCeiling() catch |e| {
        problem.* = F.explain(null, e) orelse @errorName(e);
        return error.Refused;
    };
    return .{ .window = window, .switch_name = F.ceiling_switch };
}

/// A --context past the window a family's switch raises its ceiling to: the switch named.
pub fn pastCeiling(a: Allocator, requested: i64, ceiling: i64, switch_name: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "--context {d} exceeds the {d}-token window {s} sets", .{ requested, ceiling, switch_name });
}

/// The startup line of a window past the checkpoint's trained one, which only the family's switch serves.
pub fn pastTrained(a: Allocator, window: i64, trained: i64, switch_name: []const u8) ![]const u8 {
    const words = "{s}: a {d}-token window, past the checkpoint's trained {d} tokens; past them the quality is the operator's to check";
    return std.fmt.allocPrint(a, words, .{ switch_name, window, trained });
}

/// Prompt plus reply tokens a request may use: --context (0: the model's window), else the family default within it.
pub fn contextWindow(requested: ?i64, native: i64, default: i64) error{ Negative, NoNative, PastNative }!i64 {
    const r = requested orelse return if (native > 0) @min(default, native) else default;
    if (r < 0) return error.Negative;
    if (r == 0) return if (native > 0) native else error.NoNative;
    if (native > 0 and r > native) return error.PastNative;
    return r;
}

test "context windows: the family default inside the model's, 0 for the model's, never past it" {
    try std.testing.expectEqual(@as(i64, 16384), try contextWindow(null, 262144, 16384));
    try std.testing.expectEqual(@as(i64, 4096), try contextWindow(null, 4096, 16384));
    try std.testing.expectEqual(@as(i64, 262144), try contextWindow(0, 262144, 16384));
    try std.testing.expectEqual(@as(i64, 65536), try contextWindow(65536, 262144, 16384));
    try std.testing.expectError(error.PastNative, contextWindow(300000, 262144, 16384));
    try std.testing.expectError(error.Negative, contextWindow(-5, 262144, 16384));
    try std.testing.expectError(error.NoNative, contextWindow(0, 0, 16384));
}

test "context windows: past a family's raised ceiling the refusal names its switch, and the startup line the trained window" {
    const a = std.testing.allocator;
    const past = try pastCeiling(a, 2097153, 2097152, "TF_GLM_CONTEXT_CEILING");
    defer a.free(past);
    try std.testing.expectEqualStrings("--context 2097153 exceeds the 2097152-token window TF_GLM_CONTEXT_CEILING sets", past);
    const line = try pastTrained(a, 2097152, 1048576, "TF_GLM_CONTEXT_CEILING");
    defer a.free(line);
    try std.testing.expectEqualStrings("TF_GLM_CONTEXT_CEILING: a 2097152-token window, past the checkpoint's trained 1048576 tokens; past them the quality is the operator's to check", line);
}
