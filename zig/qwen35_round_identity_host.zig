//! Native post-init ordering is executed, including real CPU YaRN resolution and identity publication.
const std = @import("std");
const Rig = @import("attribution_fake").Rig;
const t = std.testing;
fn resolvedIdentity(flag: ?f64, expected: usize) !void {
    const rig = try Rig.init();
    defer rig.deinit();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{\"max_position_embeddings\":4096,\"rope_parameters\":{\"rope_type\":\"default\"}}" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors", .data = "\x02\x00\x00\x00\x00\x00\x00\x00{}" });
    const dir = try std.fmt.allocPrint(t.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer t.allocator.free(dir);
    const engine = try @import("native_post_init").run(t.allocator, t.io, @import("native_init_fixture").Context{}, dir, .{ .rope_yarn = flag }, rig.timing);
    defer engine.deinit();
    try t.expectEqual(expected, engine.g.context);
    try t.expectEqual(@as(usize, if (flag == null) 0 else 1), engine.uploads);
    if (flag != null) {
        try t.expectEqual(@as(f64, 2), engine.uploaded.?.yarn.factor);
        try t.expectEqual(@as(usize, 4096), engine.uploaded.?.yarn.trained);
    }
    try t.expectEqual(@as(i64, @intCast(expected)), rig.timing.header.object.get("geometry").?.object.get("context").?.integer);
}
test "attribution review: resolved YaRN override publishes the installed 8192 ceiling" {
    if (comptime !@import("profile").round_attribution.enabled) return error.SkipZigTest;
    try resolvedIdentity(2, 8192);
}
test "attribution review: resolved YaRN without override retains the 4096 checkpoint ceiling" {
    if (comptime !@import("profile").round_attribution.enabled) return error.SkipZigTest;
    try resolvedIdentity(null, 4096);
}
