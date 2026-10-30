//! Refuse unload where closing the current engine also ends processes needed for reopening.
const std = @import("std");

pub fn refusal(backends: []const []const u8, model_type: []const u8, seconds: i64, world: ?[]const u8) ?[]const u8 {
    if (seconds <= 0 or !std.mem.eql(u8, model_type, "glm5_next")) return null;
    if (backends.len == 0 or !std.mem.eql(u8, backends[0], "cuda")) return null;
    const ranks = std.fmt.parseInt(usize, world orelse return null, 10) catch return null;
    if (ranks < 2) return null;
    return "--idle-unload is not supported for GLM CUDA with TF_GLM_WORLD >= 2: closing ends its followers; serve with --idle-unload 0";
}

test "idle review: GLM CUDA env ranks refuse unload independently of the CLI topology" {
    const t = std.testing;
    const args: @import("cli.zig").Args = .{ .idle_unload = 1 };
    try t.expectEqual(@as(u32, 1), args.tp);
    for ([_][]const u8{ "2", "3", "4", "8", "16" }) |world| {
        const why = refusal(&.{"cuda"}, "glm5_next", args.idle_unload, world);
        try t.expect(why != null);
        try t.expect(std.mem.indexOf(u8, why.?, "TF_GLM_WORLD") != null);
    }
    for ([_]i64{ 0, -1 }) |seconds| try t.expectEqual(null, refusal(&.{"cuda"}, "glm5_next", seconds, "4"));
    for ([_]?[]const u8{ null, "1", "invalid" }) |world| try t.expectEqual(null, refusal(&.{"cuda"}, "glm5_next", 1, world));
    try t.expectEqual(null, refusal(&.{"metal"}, "glm5_next", 1, "4"));
    try t.expectEqual(null, refusal(&.{"cuda"}, "qwen3_5", 1, "4"));
    try t.expectEqual(null, refusal(&.{}, "glm5_next", 1, "4"));
}
