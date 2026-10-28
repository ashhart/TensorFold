//! Detect DSML from a rendered assistant probe call, including whitespace around tag names and attributes.
const std = @import("std");

const mark = "<｜DSML｜";
const space = " \t\r\n";

fn tag(text: []const u8, name: []const u8) bool {
    if (!std.mem.startsWith(u8, text, name)) return false;
    return text.len > name.len and (text[name.len] == '>' or std.mem.indexOfScalar(u8, space, text[name.len]) != null);
}

pub fn isStyle(rendered: []const u8, comptime probe: []const u8, comptime key: []const u8) bool {
    var from: usize = 0;
    var in_calls = false;
    var in_probe = false;
    while (std.mem.indexOfPos(u8, rendered, from, mark)) |at| {
        const end = std.mem.indexOfScalarPos(u8, rendered, at + mark.len, '>') orelse return false;
        from = end + 1;
        const head = std.mem.trim(u8, rendered[at + mark.len .. end + 1], space);
        if (tag(head, "calls") or tag(head, "tool_calls") or tag(head, "function_calls")) {
            in_calls = true;
            in_probe = false;
        } else if (in_calls and tag(head, "invoke")) {
            in_probe = attribute(head, "name", probe);
        } else if (in_probe and tag(head, "parameter") and attribute(head, "name", key)) return true;
    }
    return false;
}

fn attribute(head: []const u8, name: []const u8, value: []const u8) bool {
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, head, from, name)) |at| {
        from = at + name.len;
        if (at == 0 or std.mem.indexOfScalar(u8, space, head[at - 1]) == null) continue;
        var rest = std.mem.trimStart(u8, head[from..], space);
        if (rest.len == 0 or rest[0] != '=') continue;
        rest = std.mem.trimStart(u8, rest[1..], space);
        if (rest.len < value.len + 2 or rest[0] != '"' or rest[value.len + 1] != '"') continue;
        if (std.mem.eql(u8, rest[1 .. value.len + 1], value)) return true;
    }
    return false;
}
