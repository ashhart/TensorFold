//! The field layer: every reader's admitted values, and each refusal by the source, field and reason it names.
const std = @import("std");
const f = @import("config_fields.zig");
const t = std.testing;

fn doc(a: std.mem.Allocator, text: []const u8) !f.Object {
    return (try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{})).object;
}

const fixture =
    \\{"hidden_size":5120,"eps":1e-06,"theta":10000000,"wide":1e999,"flag":true,"none":null,
    \\ "zero":0,"neg":-1,"big":4294967296,"huge":99999999999999999999,"text":"5","real":128.9,"list":[1]}
;

test "field layer: a fixture's admitted values, read once each" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var p: f.Problem = .{};
    const r: f.Reader = .{ .object = try doc(arena.allocator(), fixture), .source = "config.json", .problem = &p };
    try t.expectEqual(@as(usize, 5120), try r.requiredInt(usize, "hidden_size", .{ .min = 1 }));
    try t.expectEqual(@as(u32, 0), try r.requiredInt(u32, "zero", .{}));
    try t.expectEqual(@as(f64, 1e-6), try r.requiredFloat("eps"));
    try t.expectEqual(@as(f64, 10000000), try r.requiredFloat("theta")); // an integer reads as a number
    try t.expectEqual(@as(?usize, null), try r.optionalInt(usize, "absent", .refused, .{}));
    try t.expectEqual(@as(?usize, null), try r.optionalInt(usize, "none", .absent, .{}));
    try t.expectEqual(@as(?f64, null), try r.optionalFloat("none", .absent));
    try t.expectEqual(@as(?bool, true), try r.optionalFlag("flag", .refused));
    try t.expectEqual(@as(?bool, null), try r.optionalFlag("absent", .refused));
    try t.expectEqual(@as(usize, 0), p.len); // nothing refused, nothing written
}

/// One read's refusal: BadConfigField, and the Problem's exact text.
fn refused(want: []const u8, got: anytype, p: *f.Problem) !void {
    try t.expectError(error.BadConfigField, got);
    try t.expectEqualStrings(want, p.text());
    p.len = 0;
}

test "field layer: each malformed value is refused by source, field and reason; nothing is truncated or coerced" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var p: f.Problem = .{};
    const r: f.Reader = .{ .object = try doc(arena.allocator(), fixture), .source = "config.json text_config", .problem = &p };
    const u32_range: f.Ints = .{ .min = 1, .max = std.math.maxInt(u32) };
    try refused("config.json text_config: absent is missing", r.requiredInt(usize, "absent", .{}), &p);
    try refused("config.json text_config: none is null, not an integer", r.requiredInt(usize, "none", .{}), &p);
    try refused("config.json text_config: none is null", r.optionalInt(usize, "none", .refused, .{}), &p);
    try refused("config.json text_config: text is a string, not an integer", r.requiredInt(usize, "text", .{}), &p);
    try refused("config.json text_config: real is a float, not an integer", r.requiredInt(usize, "real", .{}), &p);
    try refused("config.json text_config: eps is a float, not an integer", r.optionalInt(usize, "eps", .absent, .{}), &p);
    try refused("config.json text_config: flag is a bool, not an integer", r.requiredInt(usize, "flag", .{}), &p);
    try refused("config.json text_config: list is an array, not an integer", r.requiredInt(usize, "list", .{}), &p);
    try refused("config.json text_config: huge is a number past 64 bits, not an integer", r.requiredInt(u64, "huge", .{}), &p);
    try refused("config.json text_config: zero is 0, outside 1..4294967295", r.requiredInt(usize, "zero", u32_range), &p);
    try refused("config.json text_config: neg is -1, outside 0..9223372036854775807", r.requiredInt(usize, "neg", .{}), &p);
    try refused("config.json text_config: big is 4294967296, outside 1..4294967295", r.requiredInt(usize, "big", u32_range), &p);
    try refused("config.json text_config: big is 4294967296, past u32", r.requiredInt(u32, "big", .{}), &p);
    try refused("config.json text_config: text is a string, not a number", r.requiredFloat("text"), &p);
    try refused("config.json text_config: none is null, not a number", r.requiredFloat("none"), &p);
    try refused("config.json text_config: none is null", r.optionalFloat("none", .refused), &p);
    try refused("config.json text_config: wide is a number past 64 bits, not a number", r.requiredFloat("wide"), &p);
    try refused("config.json text_config: zero is an integer, not a bool", r.optionalFlag("zero", .refused), &p);
    try refused("config.json text_config: none is null", r.optionalFlag("none", .refused), &p);
}

test "field layer: a long source and key are cut at the Problem's end, never past it" {
    var p: f.Problem = .{};
    const long: [300]u8 = @splat('k');
    try t.expectError(error.BadConfigField, f.int(usize, .{ .string = "x" }, .{}, .{ .source = &long, .key = &long }, &p));
    try t.expectEqual(p.bytes.len, p.len);
    try t.expectEqualStrings(long[0..256], p.text());
}
