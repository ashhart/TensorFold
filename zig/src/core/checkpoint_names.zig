//! Shared checkpoint index schema and portable shard-name validation.
const std = @import("std");

pub fn plainName(name: []const u8) bool {
    if (name.len == 0 or name[name.len - 1] == '.' or name[name.len - 1] == ' ') return false;
    for (name) |c| if (c < 0x20 or c == 0x7f) return false;
    if (std.mem.indexOfAny(u8, name, "/\\:<>\"|?*") != null) return false;
    const stem = name[0 .. std.mem.indexOfScalar(u8, name, '.') orelse name.len];
    for ([_][]const u8{ "CON", "PRN", "AUX", "NUL" }) |reserved|
        if (std.ascii.eqlIgnoreCase(stem, reserved)) return false;
    if (stem.len == 4 and stem[3] >= '1' and stem[3] <= '9' and
        (std.ascii.eqlIgnoreCase(stem[0..3], "COM") or std.ascii.eqlIgnoreCase(stem[0..3], "LPT"))) return false;
    return true;
}

/// An index's shard: a plain name, or plain names joined by '/' with none starting with '.', inside the model folder.
pub fn shardName(name: []const u8) bool {
    if (std.mem.indexOfScalar(u8, name, '/') == null) return plainName(name);
    var parts = std.mem.splitScalar(u8, name, '/');
    while (parts.next()) |part| if (!plainName(part) or part[0] == '.') return false;
    return true;
}

pub fn weightMap(root: std.json.Value) error{BadIndex}!std.json.ObjectMap {
    if (root != .object) return error.BadIndex;
    const map = root.object.get("weight_map") orelse return error.BadIndex;
    if (map != .object) return error.BadIndex;
    for (map.object.values()) |value| {
        if (value != .string or !shardName(value.string)) return error.BadIndex;
    }
    return map.object;
}

/// The weight map with each shard directly in the model folder, as writes need: shard_edit.linked sees only those.
pub fn plainWeightMap(root: std.json.Value) error{BadIndex}!std.json.ObjectMap {
    const map = try weightMap(root);
    for (map.values()) |value| if (!plainName(value.string)) return error.BadIndex;
    return map;
}
