const std = @import("std");

pub fn budget(context: u32, max_frames: ?u32) !struct { frames: u32, prompt: u32 } {
    if (context < 1025 or context > 8192) return error.BadDramaContext;
    if (max_frames) |n| {
        if (n < 1 or n > 8190) return error.BadDramaMaxFrames;
        if (n >= context - 1) return error.NoDramaPromptRoom;
    }
    // Keep the stock low-context clamp when the option is omitted.
    const frames = @min(max_frames orelse 1024, context - 3);
    return .{ .frames = frames, .prompt = @min(4096, context - frames - 1) };
}

test "default frame and prompt budgets match stock at every supported context" {
    for (1025..8193) |ctx| {
        const context: u32 = @intCast(ctx);
        const got = try budget(context, null);
        const stock = @min(1024, context - 3);
        try std.testing.expectEqual(stock, got.frames);
        try std.testing.expectEqual(@min(4096, context - stock - 1), got.prompt);
    }
}

test "explicit frame budgets are bounded and leave prompt room" {
    const long = try budget(8192, 4096);
    try std.testing.expectEqual(@as(u32, 4096), long.frames);
    try std.testing.expectEqual(@as(u32, 4095), long.prompt);
    const small = try budget(8192, 1);
    try std.testing.expectEqual(@as(u32, 1), small.frames);
    try std.testing.expectEqual(@as(u32, 4096), small.prompt);
    const edge = try budget(8192, 8190);
    try std.testing.expectEqual(@as(u32, 8189), edge.frames);
    try std.testing.expectEqual(@as(u32, 2), edge.prompt);
    try std.testing.expectError(error.BadDramaMaxFrames, budget(8192, 0));
    try std.testing.expectError(error.BadDramaMaxFrames, budget(8192, 8191));
    try std.testing.expectError(error.NoDramaPromptRoom, budget(4097, 4096));
    try std.testing.expectError(error.NoDramaPromptRoom, budget(1025, 1024));
    try std.testing.expectError(error.BadDramaContext, budget(1024, null));
    try std.testing.expectError(error.BadDramaContext, budget(8193, null));
    const explicit = try budget(8192, 1024);
    const default = try budget(8192, null);
    try std.testing.expectEqualDeep(default, explicit);
}
