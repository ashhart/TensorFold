const std = @import("std");
const harness = @import("speech_stream_test.zig");

test "speech admission refuses assembled rows before submitting to the engine" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const input = try a.alloc(u8, 4096);
    @memset(input, 'a');
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"input\":\"{s}\",\"response_format\":\"pcm\"}}", .{input});
    var script: harness.Script = .{ .steps = &.{}, .inline_ = true };
    var served: harness.Served = undefined;
    try served.start(&script, "HTTP/1.1", body, 2000);
    defer served.deinit();
    served.join(&script);
    const head = try served.readHead();
    try std.testing.expect(std.mem.indexOf(u8, head, "400") != null);
    while (try served.more(20)) {}
    const response = served.got.items;
    try std.testing.expect(std.mem.indexOf(u8, response, "assembled prompt rows") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "request text: 4096 tokens") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "acoustic references: 0 frames") != null);
    try std.testing.expectEqual(@as(usize, 0), script.takes);
}

test "speech admission counts multiple references and refuses before engine submission" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const input = try a.alloc(u8, 4096);
    @memset(input, 'a');
    const codes = [_][10]i32{ @splat(1), @splat(2) };
    const refs = [_]@import("drama_prompt").Reference{
        .{ .transcript = "First.", .codes = &codes },
        .{ .transcript = "Second.", .codes = codes[0..1] },
    };
    const voices = [_]@import("drama_voice.zig").Voice{.{ .name = "synthetic", .references = &refs }};
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"voice\":\"synthetic\",\"input\":\"{s}\",\"response_format\":\"pcm\"}}", .{input});
    var script: harness.Script = .{ .steps = &.{}, .inline_ = true, .voices = &voices };
    var served: harness.Served = undefined;
    try served.start(&script, "HTTP/1.1", body, 2000);
    defer served.deinit();
    served.join(&script);
    const head = try served.readHead();
    try std.testing.expect(std.mem.indexOf(u8, head, "400") != null);
    while (try served.more(20)) {}
    const response = served.got.items;
    try std.testing.expect(std.mem.indexOf(u8, response, "assembled prompt rows") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "request text: 4096 tokens") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "acoustic references: 3 frames") != null);
    try std.testing.expectEqual(@as(usize, 0), script.takes);
}
