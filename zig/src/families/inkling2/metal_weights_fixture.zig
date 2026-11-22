//! Local, opt-in original-checkpoint fixture. No test downloads or repacks the weights.
const std = @import("std");

pub fn checkpoint() ![:0]u8 {
    const t = std.testing;
    const dir = t.environ.getPosix("TF_INKLING2_MINI") orelse return error.SkipZigTest;
    return std.fmt.allocPrintSentinel(t.allocator, "{s}/model-00001-of-00001.safetensors", .{dir}, 0);
}
