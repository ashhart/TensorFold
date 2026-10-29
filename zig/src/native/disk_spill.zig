//! --spill-gib and --checkpoint-slots on any backend: the RAM store's slots beside the disk store, the refusals' words.
const std = @import("std");
const api = @import("engine_api");

/// --checkpoint-slots: the prompt states the RAM store keeps at most beside its disk store (null: the family's plan).
pub fn slots(o: api.Open) ?usize {
    return if (o.keep) |n| @intCast(@max(n, 0)) else null;
}

/// --checkpoint-slots bounds what stays in RAM beside the disk store: without one it is refused (`where`: " on Macs").
pub fn refusal(o: api.Open, comptime where: []const u8) ?[]const u8 {
    if (o.keep != null and !o.learn_conversations) return "--checkpoint-slots sets how many prompt states stay in RAM beside the disk store" ++ where ++ ": serve with --spill-gib N (or --learn-conversations), or without it";
    return null;
}

/// The flag a family without the disk store names in its refusal: --spill-gib when that alone asked for the store.
pub fn flag(o: api.Open) []const u8 {
    return if (o.spill == .evicted) "--spill-gib" else "--learn-conversations";
}

test "--checkpoint-slots takes the disk store's company, and a refusal names the flag that asked" {
    const alone: api.Open = .{ .dir = "d", .model_type = "nemotron_h", .keep = 2 };
    try std.testing.expectEqualStrings("--checkpoint-slots sets how many prompt states stay in RAM beside the disk store on Macs: serve with --spill-gib N (or --learn-conversations), or without it", refusal(alone, " on Macs") orelse "");
    try std.testing.expectStringStartsWith(refusal(alone, "") orelse "", "--checkpoint-slots sets how many prompt states stay in RAM beside the disk store: serve");
    try std.testing.expect(refusal(.{ .dir = "d", .model_type = "nemotron_h" }, "") == null);
    const spilled: api.Open = .{ .dir = "d", .model_type = "nemotron_h", .keep = 2, .learn = "/s", .learn_conversations = true, .spill = .evicted };
    try std.testing.expect(refusal(spilled, "") == null);
    try std.testing.expectEqual(@as(?usize, 2), slots(spilled));
    try std.testing.expectEqual(@as(?usize, null), slots(.{ .dir = "d", .model_type = "nemotron_h" }));
    try std.testing.expectEqualStrings("--spill-gib", flag(spilled));
    for ([_]api.prompt_cache.rows.Spill{ .off, .finish }) |m| try std.testing.expectEqualStrings("--learn-conversations", flag(.{ .dir = "d", .model_type = "t", .spill = m }));
}
