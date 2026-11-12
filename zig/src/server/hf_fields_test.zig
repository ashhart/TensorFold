//! Request fields HfText renders with: chat_template_kwargs.clear_thinking parsing and refusals.
const std = @import("std");
const json = @import("json");

test "chat_template_kwargs.clear_thinking: a bool, null as unset, anything else refused" {
    const fields = @import("fields.zig");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: @import("errors.zig").Cx = .{ .a = a };
    const cases = [_]struct { body: []const u8, want: ?bool }{
        .{ .body = "{}", .want = null },
        .{ .body = "{\"chat_template_kwargs\":{\"clear_thinking\":null}}", .want = null },
        .{ .body = "{\"chat_template_kwargs\":{\"clear_thinking\":true}}", .want = true },
        .{ .body = "{\"chat_template_kwargs\":{\"clear_thinking\":false,\"enable_thinking\":true}}", .want = false },
    };
    for (cases) |c| try std.testing.expectEqual(c.want, (try fields.thinkingFields(&cx, (try json.parse(a, c.body)).ok, &.{})).clear);
    for ([_][]const u8{ "1", "\"true\"", "{}" }) |bad| {
        const body = try std.fmt.allocPrint(a, "{{\"chat_template_kwargs\":{{\"clear_thinking\":{s}}}}}", .{bad});
        try std.testing.expectError(error.Refused, fields.thinkingFields(&cx, (try json.parse(a, body)).ok, &.{}));
        try std.testing.expectEqualStrings("chat_template_kwargs.clear_thinking must be true or false", cx.message);
    }
}
