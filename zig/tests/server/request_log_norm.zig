//! Request-log normalisation for the frozen golden check.
const std = @import("std");

/// Drop reply lines and the random request id so two logs compare byte-exactly on request bodies.
pub fn requestOnly(a: std.mem.Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        var parsed = try std.json.parseFromSlice(std.json.Value, a, line, .{});
        defer parsed.deinit();
        var v = parsed.value; // var: the id is removed from this copy before it is written out
        if (v == .object) {
            if (v.object.get("kind")) |k| {
                if (k == .string and std.mem.eql(u8, k.string, "reply")) continue;
            }
            _ = v.object.orderedRemove("id");
        }
        const normalized = try std.json.Stringify.valueAlloc(a, v, .{});
        try out.appendSlice(a, normalized);
        try out.append(a, '\n');
    }
    return out.items;
}
