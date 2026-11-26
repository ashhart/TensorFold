const std = @import("std");
const chat = @import("chat_cli.zig");
const stream = @import("benchmark_stream.zig");

test "chat CLI: existing keys retain main's non-Latin-1 acceptance and bounded containment" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var env: std.process.Environ.Map = .init(a);
    const key = "fixture-\u{20ac}";
    try env.put("TENSORFOLD_API_KEY", key);
    try std.testing.expectEqualStrings(key, (try chat.envKey(&env)).?);
    var buf: [300]u8 = undefined;
    try std.testing.expectEqualStrings("Bearer " ++ key, stream.rawHeader(&buf, key).?);
    const credentials = try @import("chat_cli_turn.zig").Credentials.init(a, &.{key});
    const safe = (try credentials.safe(a, "server reflected fixture-\u{20ac}")).?;
    try std.testing.expect(std.mem.indexOf(u8, safe, key) == null);
    const long: [257]u8 = @splat('k');
    try env.put("TENSORFOLD_API_KEY", &long);
    try std.testing.expectError(error.Refused, chat.envKey(&env));
}

test "chat CLI: serve registration keeps main's long, non-Latin-1 and legacy-control acceptance" {
    const long: [300]u8 = @splat('k');
    var problem: []const u8 = "";
    var store = try @import("auth.zig").Store.init(std.testing.allocator, &.{ &long, "fixture-\u{20ac}", "fixture-\u{007f}", "fixture-\u{009b}" }, "", null, false, &problem);
    defer store.deinit();
    try std.testing.expectEqual(@as(usize, 4), store.static.items.len);
    try std.testing.expectEqualStrings("cli-1", store.static.items[0].label);
    try std.testing.expectEqualStrings("cli-2", store.static.items[1].label);
    try std.testing.expectEqualStrings("cli-3", store.static.items[2].label);
    try std.testing.expectEqualStrings("cli-4", store.static.items[3].label);
}
