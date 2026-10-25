//! Named config refusals, with no process-global diagnostic or borrowed parse storage.
const std = @import("std");
pub const Object = std.json.ObjectMap;
pub const Problem = struct {
    bytes: [256]u8 = undefined,
    len: usize = 0,
    pub fn name(p: *const Problem) []const u8 {
        return p.bytes[0..p.len];
    }
    pub fn reject(p: *Problem, field: []const u8) error{UnsupportedQwen4} {
        p.len = @min(field.len, p.bytes.len);
        @memcpy(p.bytes[0..p.len], field[0..p.len]);
        return error.UnsupportedQwen4;
    }
};
pub fn object(v: std.json.Value, field: []const u8, p: *Problem) !Object {
    return if (v == .object) v.object else p.reject(field);
}
pub fn int(o: Object, field: []const u8, p: *Problem) !usize {
    const v = o.get(field) orelse return p.reject(field);
    if (v != .integer or v.integer <= 0) return p.reject(field);
    return std.math.cast(usize, v.integer) orelse p.reject(field);
}
pub fn natural(o: Object, field: []const u8, default: usize, p: *Problem) !usize {
    const v = o.get(field) orelse return default;
    if (v == .null) return default;
    if (v != .integer or v.integer < 0) return p.reject(field);
    return std.math.cast(usize, v.integer) orelse p.reject(field);
}
pub fn number(v: std.json.Value, field: []const u8, p: *Problem) !f64 {
    const x: f64 = switch (v) {
        .float => |x| x,
        .integer => |x| @floatFromInt(x),
        else => return p.reject(field),
    };
    return if (std.math.isFinite(x)) x else p.reject(field);
}
pub fn word(v: std.json.Value, want: []const u8) bool {
    return v == .string and std.mem.eql(u8, v.string, want);
}
pub fn flag(o: Object, field: []const u8, default: bool, p: *Problem) !bool {
    const v = o.get(field) orelse return default;
    return if (v == .bool) v.bool else p.reject(field);
}
pub fn fixed(o: Object, field: []const u8, want: std.json.Value, p: *Problem) !void {
    const v = o.get(field) orelse return;
    const ok = switch (want) {
        .string => |s| word(v, s),
        .bool => |b| v == .bool and v.bool == b,
        .integer => |i| v == .integer and v.integer == i,
        else => false,
    };
    if (!ok) return p.reject(field);
}

/// Unknown config keys can alter an architecture. Only explicit metadata is ignored.
pub fn known(o: Object, names: []const []const u8, p: *Problem) !void {
    for (o.keys()) |key| {
        for (names) |name| {
            if (std.mem.eql(u8, key, name)) break;
        } else return p.reject(key);
    }
}
