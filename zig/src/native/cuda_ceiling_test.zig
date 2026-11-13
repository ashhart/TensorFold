//! A family's raised context ceiling (GLM's TF_GLM_CONTEXT_CEILING) through the real CUDA open, before any weights.
const std = @import("std");
const env = @import("sys").env;
const cuda = @import("cuda.zig");

test "context ceilings: GLM reads its ceiling at its open, unset keeps main's refusal, and a malformed one is GLM's refusal" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"max_position_embeddings\": 1048576}" });
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    var named: usize = 0;
    inline for (cuda.registry) |F| if (comptime @hasDecl(F, "raisedCeiling")) {
        defer _ = env.unsetenv(F.ceiling_switch);
        var why: []const u8 = "";
        _ = env.unsetenv(F.ceiling_switch);
        try std.testing.expect((try cuda.open(a, std.testing.allocator, std.testing.io, .{ .dir = dir, .model_type = F.model_type, .context = 2097152 }, &why)) == null);
        try std.testing.expectEqualStrings("--context 2097152 exceeds this model's 1048576-token window", why);
        _ = env.setenv(F.ceiling_switch, "2M", 1);
        try std.testing.expect((try cuda.open(a, std.testing.allocator, std.testing.io, .{ .dir = dir, .model_type = F.model_type, .context = 2097152 }, &why)) == null);
        try std.testing.expectEqualStrings(F.explain(null, error.InvalidEnvironment).?, why);
        _ = env.setenv(F.ceiling_switch, "2097152", 1);
        try std.testing.expect((try cuda.open(a, std.testing.allocator, std.testing.io, .{ .dir = dir, .model_type = F.model_type, .context = 2097153 }, &why)) == null);
        try std.testing.expectEqualStrings("--context 2097153 exceeds the 2097152-token window " ++ F.ceiling_switch ++ " sets", why);
        named += 1;
    };
    try std.testing.expect(named >= 1); // GLM-5.3-Flash's TF_GLM_CONTEXT_CEILING
}
