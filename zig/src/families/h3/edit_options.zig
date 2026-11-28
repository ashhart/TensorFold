//! tf-h3-image-prompt's edit options, independent of its Metal input readers.
const std = @import("std");

/// Each `--reference FILE` and `--anchors PREFIX` on the command line, in order.
pub const Options = struct { references: []const []const u8 = &.{}, anchors: []const []const u8 = &.{} };

/// Parse `args`; error.BadArguments for a missing value or repeated references.
pub fn take(a: std.mem.Allocator, args: []const []const u8) !struct { []const []const u8, Options } {
    var rest: std.ArrayList([]const u8) = .empty;
    var refs: std.ArrayList([]const u8) = .empty;
    var prefixes: std.ArrayList([]const u8) = .empty;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const list = if (std.mem.eql(u8, args[i], "--reference")) &refs else if (std.mem.eql(u8, args[i], "--anchors")) &prefixes else {
            try rest.append(a, args[i]);
            continue;
        };
        i += 1;
        if (i == args.len) return error.BadArguments;
        try list.append(a, args[i]);
    }
    if (refs.items.len > 1) return error.BadArguments;
    return .{ rest.items, .{ .references = refs.items, .anchors = prefixes.items } };
}

test "h3 image edit options reject multiple references but keep zero or one" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const none_rest, const none = try take(a, &.{ "tf-h3-image-prompt", "model", "prompt" });
    try std.testing.expectEqual(@as(usize, 0), none.references.len);
    try std.testing.expectEqual(@as(usize, 0), none.anchors.len);
    try std.testing.expectEqualSlices([]const u8, &.{ "tf-h3-image-prompt", "model", "prompt" }, none_rest);

    const one_rest, const one = try take(a, &.{ "tf-h3-image-prompt", "--reference", "vision", "--anchors", "latent", "prompt" });
    try std.testing.expectEqualSlices([]const u8, &.{"vision"}, one.references);
    try std.testing.expectEqualSlices([]const u8, &.{"latent"}, one.anchors);
    try std.testing.expectEqualSlices([]const u8, &.{ "tf-h3-image-prompt", "prompt" }, one_rest);
    try std.testing.expectError(error.BadArguments, take(a, &.{ "--reference", "one", "--anchors", "first", "--reference", "two", "--anchors", "second" }));
}
