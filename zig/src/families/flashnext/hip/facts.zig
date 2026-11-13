//! The lane core's facts of a loaded Flash Next model, pure so a host test drives them without a device.
const std = @import("std");
const lanes = @import("lanes");
const cache = @import("cache.zig");

/// MTP drafts the HIP head takes at most: `--mtp-drafts` is refused past it (native/hip.zig's check).
pub const mtp_max_drafts: u32 = 3;

/// The policy's `--mtp-drafts` capped at the head's.
pub fn capDrafts(asked: u8) u32 {
    return @min(mtp_max_drafts, @as(u32, asked));
}

/// A shared round's copy window and a lone stream's first: the pending row and three copies; lone runs grow past it.
const shared_copy_rows = 4;

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
