//! Actual native startup refuses a narrow environment before the loader; a missing fixture cannot open a GPU.
const std = @import("std");
const host = @import("qwen27_served");
const tf = @import("tensorfold");
const t = std.testing;
const diagnostics = @import("qwen_weights_diagnostics");
test "qwen27 rows entry: startup names the supplied setting before a loader failure" {
    const value = std.mem.span(std.c.getenv("TF_PROMPT_ROWS").?);
    const count = try std.fmt.parseInt(u32, value, 10);
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(t.allocator, ".zig-cache/tmp/{s}/missing", .{tmp.sub_path});
    defer t.allocator.free(path);
    var why: []const u8 = "";
    defer if (why.len > 0) t.allocator.free(why);
    const opened = host.open(t.allocator, t.io, path, .{ .context = 1024, .streams = 1 }, t.allocator, &why);
    if (count < 128) {
        try t.expectError(error.PromptRowsTooSmall, opened);
        const given = try std.fmt.allocPrint(t.allocator, "TF_PROMPT_ROWS supplied {d}.", .{count});
        defer t.allocator.free(given);
        try t.expect(std.mem.indexOf(u8, why, "at least 128 prompt rows") != null);
        try t.expect(std.mem.indexOf(u8, why, given) != null);
        try t.expect(std.mem.indexOf(u8, why, "unset it") != null);
    } else {
        try t.expectError(error.FileNotFound, opened);
        try t.expectEqualStrings("", why);
    }
}
test "qwen27 rows entry: the actual lane constructor names a caller-created narrow frame" {
    var model: tf.qwen27.model.Model = undefined;
    model.frame.capacity = 64;
    var runner: tf.qwen27.decode_round.Runner = undefined;
    runner.slots = 1;
    runner.model = &model;
    try t.expectError(error.PromptRowsTooSmall, host.lanes_backend.Metal.init(t.allocator, &runner, null));
    try t.expect(std.mem.indexOf(u8, diagnostics.message, "model.frame.capacity supplied 64") != null);
    try t.expect(std.mem.indexOf(u8, diagnostics.message, "at least 128 prompt rows") != null);
    try diagnostics.expect(1);
}
