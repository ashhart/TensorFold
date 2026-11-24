//! Voice catalogs on the wire: reference rows and the HTTP listing, against speech_stream_test.zig's scripted engine.
const std = @import("std");
const harness = @import("speech_stream_test.zig");
const Script = harness.Script;
const Served = harness.Served;
const Bytes = harness.Bytes;
const frame = harness.frame;
const pcm = harness.pcm;
const expectHeader = harness.expectHeader;
const gpa = std.testing.allocator;

test "voices: HTTP listing and reference rows use the family CLI builder" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const family = @import("drama_prompt");
    const codes = [_][10]i32{.{ 7, 1, 2, 3, 4, 5, 6, 7, 8, 9 }};
    const refs = [_]family.Reference{.{ .transcript = "A small cloud passes.", .codes = &codes }};
    const voices = [_]@import("drama_voice.zig").Voice{.{ .name = "tone", .references = &refs }};
    const CliTokenizer = struct {
        pub fn specialTokenId(_: @This(), _: []const u8) ?u32 {
            return 1000;
        }
        pub fn encode(_: @This(), alloc: std.mem.Allocator, input: []const u8) ![]u32 {
            var unused: u8 = 0;
            return Bytes.encode(&unused, alloc, input, true);
        }
    };
    const expected = try family.build(a, CliTokenizer{}, "hi", &refs);
    var script: Script = .{ .steps = &.{frame}, .voices = &voices, .expected_rows = expected.frames };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", "{\"model\":\"m\",\"input\":\"hi\",\"voice\":\"tone\",\"response_format\":\"pcm\"}", 2000);
    defer s.deinit();
    try expectHeader(try s.readHead(), " 200 ");
    s.join(&script);
    try std.testing.expect(script.rows_equal);
    try std.testing.expectEqual(expected.reference_rows, script.reference_rows[0]);
    try std.testing.expectEqualStrings(&pcm(1), try s.readRest());
    var listing_script: Script = .{ .steps = &.{}, .voices = &voices, .get_voices = true };
    var listing: Served = undefined;
    try listing.start(&listing_script, "HTTP/1.1", "", 2000);
    defer listing.deinit();
    try expectHeader(try listing.readHead(), " 200 ");
    listing.join(&listing_script);
    try std.testing.expectEqualStrings("{\"voices\": [\"default\", \"tone\"], \"details\": [{\"id\": \"tone\", \"description\": \"\"}]}", try listing.readRest());
}

test "voices: without configuration the default prompt rows stay unchanged" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var bytes: Bytes = .{};
    const tokens = try @import("drama_prompt.zig").encode(a, bytes.text(), "hi");
    const rows = try a.alloc([11]i32, tokens.len);
    for (rows, tokens) |*row, token| {
        row.* = @splat(0);
        row[0] = @intCast(token);
    }
    var script: Script = .{ .steps = &.{frame}, .expected_rows = rows };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\"}", 2000);
    defer s.deinit();
    try expectHeader(try s.readHead(), " 200 ");
    s.join(&script);
    try std.testing.expect(script.rows_equal);
    try std.testing.expectEqual(@as(u32, 0), script.reference_rows[0]);
    try std.testing.expectEqualStrings(&pcm(1), try s.readRest());
}
