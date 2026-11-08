//! Query parameters: first match, form decoding, malformed escapes and pydantic booleans.
const std = @import("std");
const query = @import("query.zig");
const t = std.testing;

test "ocr query: the first match after the question mark" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const path = "/ocr?det_arch=db_resnet50&resolve_blocks=true&resolve_blocks=false&flag&empty=";
    try t.expectEqualStrings("true", (try query.get(a, path, "resolve_blocks")).?);
    try t.expectEqualStrings("db_resnet50", (try query.get(a, path, "det_arch")).?);
    try t.expectEqualStrings("", (try query.get(a, path, "flag")).?);
    try t.expectEqualStrings("", (try query.get(a, path, "empty")).?);
    try t.expect(try query.get(a, path, "missing") == null);
    try t.expect(try query.get(a, path, "resolve") == null);
    try t.expect(try query.get(a, "/ocr", "format") == null);
    try t.expect(try query.get(a, "/ocr?", "format") == null);
    try t.expectEqualStrings("doctr", (try query.get(a, "/ocr/?&&format=doctr", "format")).?);
}

test "ocr query: plus and percent escapes decode in names and values" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const path = "/ocr?a+b=c+d&path=x%2Fy%2fz&n%61me=%41%20B&raw=plain";
    try t.expectEqualStrings("c d", (try query.get(a, path, "a b")).?);
    try t.expectEqualStrings("x/y/z", (try query.get(a, path, "path")).?);
    try t.expectEqualStrings("A B", (try query.get(a, path, "name")).?);
    const raw = (try query.get(a, path, "raw")).?;
    try t.expect(@intFromPtr(raw.ptr) == @intFromPtr(path.ptr) + path.len - "plain".len);
}

test "ocr query: a malformed escape in the matched value is malformed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try t.expectError(error.Malformed, query.get(a, "/ocr?x=%G1", "x"));
    try t.expectError(error.Malformed, query.get(a, "/ocr?x=ab%4", "x"));
    try t.expectError(error.Malformed, query.get(a, "/ocr?x=%", "x"));
    try t.expectError(error.Malformed, query.get(a, "/ocr?x=%G1&x=1", "x"));
    try t.expectEqualStrings("1", (try query.get(a, "/ocr?y=%G1&x=1", "x")).?);
    try t.expect(try query.get(a, "/ocr?x%G1=1", "x") == null);
}

test "ocr query: pydantic boolean words" {
    const yes = [_][]const u8{ "true", "TRUE", "True", "t", "T", "1", "yes", "YeS", "y", "on", "ON" };
    const no = [_][]const u8{ "false", "False", "f", "F", "0", "no", "NO", "n", "off", "Off" };
    const neither = [_][]const u8{ "", "2", "tru", "truee", " true", "maybe", "none", "null" };
    for (yes) |w| try t.expectEqual(true, query.boolean(w).?);
    for (no) |w| try t.expectEqual(false, query.boolean(w).?);
    for (neither) |w| try t.expect(query.boolean(w) == null);
}
