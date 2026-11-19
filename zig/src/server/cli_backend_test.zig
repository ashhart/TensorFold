//! The --backend flag's choice list versus the build's served gate, apart from cli_test.zig's 600-line cap.
const std = @import("std");
const cli = @import("cli.zig");
const Usage = cli.Usage;
const parse = cli.parse;
const servedBy = cli.Test.servedBy;

test "flag deps: --backend vulkan parses where served and gets the gated refusal elsewhere" {
    const backend_values = cli.Test.backend_values;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    if (servedBy(backend_values, "vulkan")) {
        const parsed = try parse(a, &.{ "m", "--backend", "vulkan" }, &u);
        try std.testing.expectEqualStrings("vulkan", parsed.backend);
    } else {
        try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--backend", "vulkan" }, &u));
        try std.testing.expect(std.mem.indexOf(u8, u.message, "not served by the native engine") != null);
    }
}
