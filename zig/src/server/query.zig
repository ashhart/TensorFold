//! Query parameters of a request path, decoded as form fields are (`+` is a space, `%XX` a byte).
const std = @import("std");
const Allocator = std.mem.Allocator;

/// The first value of `name` after the `?` (`""` without `=`), or null; a bad `%` escape in it is malformed.
pub fn get(a: Allocator, raw_path: []const u8, name: []const u8) error{ Malformed, OutOfMemory }!?[]const u8 {
    const q = std.mem.indexOfScalar(u8, raw_path, '?') orelse return null;
    var it = std.mem.splitScalar(u8, raw_path[q + 1 ..], '&');
    while (it.next()) |field| {
        const eq = std.mem.indexOfScalar(u8, field, '=') orelse field.len;
        if (!named(field[0..eq], name)) continue;
        return try decode(a, if (eq == field.len) "" else field[eq + 1 ..]);
    }
    return null;
}

/// A query word as pydantic reads a boolean (`true/t/yes/y/on/1` and `false/f/no/n/off/0`, any case), else null.
pub fn boolean(text: []const u8) ?bool {
    for ([_][]const u8{ "true", "t", "yes", "y", "on", "1" }) |word| if (std.ascii.eqlIgnoreCase(text, word)) return true;
    for ([_][]const u8{ "false", "f", "no", "n", "off", "0" }) |word| if (std.ascii.eqlIgnoreCase(text, word)) return false;
    return null;
}

/// Whether an encoded key decodes to `name`, without allocating; a key with a malformed escape never matches.
fn named(key: []const u8, name: []const u8) bool {
    var i: usize = 0;
    var n: usize = 0;
    while (i < key.len) : (n += 1) {
        const ch = (byte(key, &i) catch return false);
        if (n == name.len or name[n] != ch) return false;
    }
    return n == name.len;
}

/// The byte at `i.*`, decoded, with `i.*` moved past it.
fn byte(text: []const u8, i: *usize) error{Malformed}!u8 {
    const ch = text[i.*];
    if (ch == '+') {
        i.* += 1;
        return ' ';
    }
    if (ch != '%') {
        i.* += 1;
        return ch;
    }
    if (i.* + 2 >= text.len) return error.Malformed;
    const hi = std.fmt.charToDigit(text[i.* + 1], 16) catch return error.Malformed;
    const lo = std.fmt.charToDigit(text[i.* + 2], 16) catch return error.Malformed;
    i.* += 3;
    return hi * 16 + lo;
}

/// The decoded value: a slice of `text` when nothing is escaped, else a copy from `a`.
fn decode(a: Allocator, text: []const u8) error{ Malformed, OutOfMemory }![]const u8 {
    if (std.mem.indexOfAny(u8, text, "+%") == null) return text;
    var out = try std.ArrayList(u8).initCapacity(a, text.len);
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < text.len) out.appendAssumeCapacity(try byte(text, &i));
    return out.toOwnedSlice(a);
}
