//! The lane core's facts of a loaded Flash Next model, pure so a host test drives them without a device.
const std = @import("std");
const lanes = @import("lanes");
const hip = @import("hip");
const cache = @import("cache.zig");

/// A shared round's copy window and a lone stream's first: the pending row and three copies; lone runs grow past it.
const shared_copy_rows = 4;

/// Most head drafts a stream chains in a round it shares; a lone stream chains the policy's drafts.
pub const shared_chain_drafts = 3;

/// The lane core's facts of a loaded model: `drafting` names a head that drafts, `drafts` its capped chains.
pub fn factsFor(exact_width: u32, streams: usize, drafting: bool, drafts: u32) lanes.Model {
    return .{
        .exact_width = exact_width,
        .first_copy_rows = shared_copy_rows,
        .gpu_tokens = false,
        .mtp = drafting,
        .speculate = drafting,
        .speculate_early = false,
        .drafts = drafts,
        .shared_mtp_cap = shared_chain_drafts,
        .hidden_rows = true,
        .batch_rows = @intCast(exact_width * streams),
        .max_streams = @intCast(streams),
        .draft_streams = false,
    };
}

test "the shared round's copy window: the lane core takes three copies, a lone stream the whole window" {
    const a = std.testing.allocator;
    var cfg = try lanes.Config.init(a, factsFor(cache.max_window, 2, true, 3), cache.max_window, cache.max_window - 1);
    defer cfg.deinit(a);
    try std.testing.expectEqual(@as(u32, 3), cfg.first_copy);
    try std.testing.expectEqual(@as(u32, 7), cfg.max_copy);
    try std.testing.expectEqual(@as(u32, 4), cfg.base_width);
}

test "the facts cap shared chains at three, and the default policy asks the lane core for three drafts" {
    const a = std.testing.allocator;
    const asked: u32 = (hip.Policy{}).mtp.drafts;
    var cfg = try lanes.Config.init(a, factsFor(cache.max_window, 1, true, asked), cache.max_window, cache.max_window - 1);
    defer cfg.deinit(a);
    try std.testing.expectEqual(@as(u32, 3), cfg.most_drafts);
    try std.testing.expectEqual(@as(u32, 3), cfg.shared_mtp_cap);
    var deep = try lanes.Config.init(a, factsFor(cache.max_window, 1, true, 7), cache.max_window, cache.max_window - 1);
    defer deep.deinit(a);
    try std.testing.expectEqual(@as(u32, 7), deep.most_drafts);
    try std.testing.expectEqual(@as(u32, 3), deep.shared_mtp_cap);
}
