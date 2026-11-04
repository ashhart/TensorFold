const std = @import("std");

pub fn budget(context: u32, max_frames: ?u32) !struct { frames: u32, prompt: u32 } {
    if (context < 1025 or context > 8192) return error.BadDramaContext;
    if (max_frames) |n| {
        if (n < 1 or n > 8190) return error.BadDramaMaxFrames;
        if (n > context - 3) return error.NoDramaPromptRoom;
    }
    // Keep the stock low-context clamp when the option is omitted.
    const frames = max_frames orelse @min(1024, context - 3);
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

test "explicit frame budgets are exact at every supported context boundary" {
    for (1025..8193) |ctx| {
        const context: u32 = @intCast(ctx);
        const frames = context - 3;
        const got = try budget(context, frames);
        try std.testing.expectEqual(frames, got.frames);
        try std.testing.expectEqual(@as(u32, 2), got.prompt);
        try std.testing.expectError(error.NoDramaPromptRoom, budget(context, frames + 1));
    }
}

test "explicit frame budgets are bounded and leave prompt room" {
    const long = try budget(8192, 4096);
    try std.testing.expectEqual(@as(u32, 4096), long.frames);
    try std.testing.expectEqual(@as(u32, 4095), long.prompt);
    const small = try budget(8192, 1);
    try std.testing.expectEqual(@as(u32, 1), small.frames);
    try std.testing.expectEqual(@as(u32, 4096), small.prompt);
    try std.testing.expectError(error.NoDramaPromptRoom, budget(8192, 8190));
    const edge = try budget(8192, 8189);
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

pub fn metalShapeProblem(window: i64, tp: usize, rank: usize, lanes: usize) ?[]const u8 {
    if (window < 1025 or window > 8192)
        return "Drama 3 Metal supports --context 1025 through 8192 positional rows, including prompt, generated frames and one reserved row";
    if (tp != 1 or rank != 0)
        return "Drama 3 Metal serves one device per engine; tensor parallelism is unsupported (--tp 1, --rank 0)";
    if (lanes == 0 or lanes > 8)
        return "Drama 3 Metal supports 1 through 8 speech frame lanes";
    return null;
}

test "Drama Metal shape refusals preserve the qualified window and single device" {
    const t = std.testing;
    for ([_]i64{ 1025, 4096, 8192 }) |window| {
        try t.expectEqual(null, metalShapeProblem(window, 1, 0, 1));
        try t.expectEqual(null, metalShapeProblem(window, 1, 0, 8));
    }
    for ([_]i64{ -1, 0, 1024, 8193, 262144 }) |window|
        try t.expect(std.mem.indexOf(u8, metalShapeProblem(window, 1, 0, 1).?, "positional rows") != null);
    try t.expect(std.mem.indexOf(u8, metalShapeProblem(8192, 2, 0, 1).?, "tensor parallelism is unsupported") != null);
    try t.expect(std.mem.indexOf(u8, metalShapeProblem(8192, 1, 1, 1).?, "tensor parallelism is unsupported") != null);
    for ([_]usize{ 0, 9 }) |lanes|
        try t.expect(std.mem.indexOf(u8, metalShapeProblem(8192, 1, 0, lanes).?, "speech frame lanes") != null);
}
