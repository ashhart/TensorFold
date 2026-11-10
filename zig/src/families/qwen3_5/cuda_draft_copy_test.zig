//! The dense 27B's copy match beside each drafter kind and without one (draft.copyMatch).
const std = @import("std");
const draft = @import("cuda_draft.zig");

test "Qwen copy match: the dense 27B copies after 12 tokens beside DSpine, 20 beside DFlash2, 5 without; others keep the target's" {
    const dense: struct { hidden: usize = 5120, layers: usize = 64, experts: usize = 0 } = .{};
    const moe: struct { hidden: usize = 2048, layers: usize = 40, experts: usize = 256 } = .{};
    try std.testing.expectEqual(@as(i64, 12), draft.copyMatch(dense, .dspine));
    try std.testing.expectEqual(@as(i64, 20), draft.copyMatch(dense, .dflash2));
    // 0: the drafter leaves the target's copy match
    try std.testing.expectEqual(@as(i64, 0), draft.copyMatch(moe, .dspine));
    try std.testing.expectEqual(@as(i64, 0), draft.copyMatch(moe, .dflash2));
    try std.testing.expectEqual(@as(i64, 5), @import("cuda_native.zig").laneFacts(16, 64).copy_match); // no drafter
    try std.testing.expectEqual(@as(i64, 20), @import("lanes").windows.long_match); // the copy prior's threshold stays
}

test "Qwen copy match: a drafter's facts use its own kind" {
    const G = struct { hidden: usize = 5120, layers: usize = 64, experts: usize = 0 };
    const e: struct { g: G = .{} } = .{};
    const D = struct { e: *const @TypeOf(e), c: struct { kind: draft.Config.Kind } };
    const spine: D = .{ .e = &e, .c = .{ .kind = .dspine } };
    const flash: D = .{ .e = &e, .c = .{ .kind = .dflash2 } };
    try std.testing.expectEqual(@as(i64, 12), draft.factsCopyMatch(spine));
    try std.testing.expectEqual(@as(i64, 20), draft.factsCopyMatch(flash));
}
