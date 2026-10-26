//! --spill-gib and --checkpoint-slots on Macs: what the disk store's families take, and the refusal without one.
const std = @import("std");
const api = @import("engine_api");
const neutral = @import("disk_spill.zig"); // CUDA's too

/// --checkpoint-slots: the prompt states the RAM store keeps at most beside its disk store (null: the family's plan).
pub fn slots(o: api.Open) ?usize {
    return neutral.slots(o);
}

/// --checkpoint-slots bounds what stays in RAM beside the disk store, so a Mac refuses it without one.
pub fn refusal(o: api.Open) ?[]const u8 {
    return neutral.refusal(o, " on Macs");
}

/// Whether a disk store writes states behind (--learn, --learn-conversations), not only those RAM evicts (--spill-gib).
pub fn writesBehind(o: api.Open) bool {
    return o.learn != null and o.spill != .evicted;
}

/// The flag a family without the disk store names in its refusal: --spill-gib when that alone asked for the store.
pub fn flag(o: api.Open) []const u8 {
    return neutral.flag(o);
}

test "--checkpoint-slots on a Mac takes the disk store's company, and a refusal names the flag that asked" {
    const alone: api.Open = .{ .dir = "d", .model_type = "nemotron_h", .keep = 2 };
    try std.testing.expectStringStartsWith(refusal(alone) orelse "", "--checkpoint-slots sets how many");
    try std.testing.expect(refusal(.{ .dir = "d", .model_type = "nemotron_h" }) == null);
    const spilled: api.Open = .{ .dir = "d", .model_type = "nemotron_h", .keep = 2, .learn = "/s", .learn_conversations = true, .spill = .evicted };
    try std.testing.expect(refusal(spilled) == null);
    try std.testing.expectEqual(@as(?usize, 2), slots(spilled));
    try std.testing.expectEqual(@as(?usize, null), slots(.{ .dir = "d", .model_type = "nemotron_h" }));
    try std.testing.expectEqualStrings("--spill-gib", flag(spilled));
    for ([_]api.prompt_cache.rows.Spill{ .off, .finish }) |m| try std.testing.expectEqualStrings("--learn-conversations", flag(.{ .dir = "d", .model_type = "t", .spill = m }));
}

test "--spill-gib alone writes nothing behind, so Flash Next's shared sessions may open with it" {
    try std.testing.expect(!writesBehind(.{ .dir = "d", .model_type = "qwen4_exp" }));
    try std.testing.expect(!writesBehind(.{ .dir = "d", .model_type = "qwen4_exp", .learn = "/s", .learn_conversations = true, .spill = .evicted }));
    try std.testing.expect(writesBehind(.{ .dir = "d", .model_type = "qwen4_exp", .learn = "/s", .learn_conversations = true, .spill = .finish }));
    try std.testing.expect(writesBehind(.{ .dir = "d", .model_type = "qwen4_exp", .learn = "/s" }));
}
