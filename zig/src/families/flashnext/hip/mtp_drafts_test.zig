//! The HIP head's MTP draft cap: the window's seven at most, the policy's default three.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const cache = @import("cache.zig");
const facts_mod = @import("facts.zig");
const fx = @import("lanes.zig");

test "MTP drafts reach the window's seven, and the policy's default stays three" {
    try std.testing.expectEqual(@as(u32, 7), fx.mtp_max_drafts);
    try std.testing.expectEqual(@as(u32, 3), fx.capDrafts((hip.Policy{}).mtp.drafts));
    try std.testing.expectEqual(@as(u32, 7), fx.capDrafts(200));
    const m = facts_mod.factsFor(cache.max_window, 2, true, fx.capDrafts(7));
    try std.testing.expectEqual(@as(u32, 7), m.drafts);
    var cfg = try lanes.Config.init(std.testing.allocator, m, cache.max_window, cache.max_window - 1);
    defer cfg.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 7), cfg.most_drafts);
}
