//! CLI defaults name every active request effort; thinking off remains the separate existing flag.
const std = @import("std");
const cli = @import("cli.zig");
const fields = @import("fields.zig");

test "capability audit: effort defaults and CLI discovery use every active request effort" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var usage: cli.Usage = .{};
    var output: std.Io.Writer.Allocating = .init(a);
    try cli.capabilities(&output.writer, .{ .version = "test" });
    const document = (try std.json.parseFromSlice(std.json.Value, a, output.written(), .{})).value;
    const names = document.object.get("serve").?.object.get("flags").?.object.get("--reasoning-effort").?.object.get("values").?.array.items;
    try std.testing.expectEqual(fields.efforts.len - 1, names.len);
    for (fields.efforts[1..], names) |effort, name| {
        try std.testing.expectEqualStrings(effort, name.string);
        const args = try cli.parse(a, &.{ "m", "--reasoning-effort", effort }, &usage);
        try std.testing.expectEqualStrings(effort, args.reasoning_effort.?);
    }
    try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--reasoning-effort", "none" }, &usage));
    try std.testing.expect(!(try cli.parse(a, &.{ "m", "--no-thinking" }, &usage)).thinking);
}
