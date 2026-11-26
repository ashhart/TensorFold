//! Legacy chat preserves the original key bytes on discovery and streamed turns.
const std = @import("std");
const fake = @import("benchmark_fake.zig");
const fixtures = @import("chat_cli_test.zig");

test "chat CLI: every accepted legacy key keeps main's exact discovery and POST bearer bytes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try fixtures.Run.init(a, &tmp);
    const replies = [_]fake.Write{
        .{ .bytes = try fake.headOf(a, "200 OK", "text/event-stream") },
        .{ .bytes = try fake.event(a, "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Ready.\"},\"finish_reason\":\"stop\"}]}") },
        .{ .bytes = try fake.event(a, "[DONE]") },
    };
    const response = [_][]const fake.Write{&replies};
    const f = try fixtures.world(a, fixtures.models_doc, &response);
    defer f.stop();
    var long: [256]u8 = undefined;
    @memset(&long, 'k');
    const keys = [_][]const u8{ "fixture-caf\u{e9}", "fixture-\u{20ac}", "fixture-\u{009b}", "fixture-\xff", &long };
    r.input = "hi";
    for (keys) |key| {
        try r.env.put("TENSORFOLD_API_KEY", key);
        const before = f.count();
        try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "--url", try f.url(a, "") }));
        try std.testing.expectEqual(before + 2, f.count());
        const expected = try std.mem.concat(a, u8, &.{ "Bearer ", key });
        try std.testing.expectEqualSlices(u8, expected, f.header(before, "authorization").?);
        try std.testing.expectEqualSlices(u8, expected, f.header(before + 1, "authorization").?);
        try std.testing.expect(std.mem.indexOf(u8, r.err.written(), key) == null);
    }
}
