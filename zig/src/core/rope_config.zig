//! A tower's RoPE keys, read one way: `rope_parameters`, the legacy `rope_scaling`, and the top-level keys behind them.
const std = @import("std");
const ObjectMap = std.json.ObjectMap;

pub const Error = error{BadRopeConfig};

/// `rope_parameters`: null when absent or null; anything but an object is refused.
pub fn params(tower: ObjectMap) Error!?ObjectMap {
    const v = tower.get("rope_parameters") orelse return null;
    return switch (v) {
        .null => null,
        .object => |o| o,
        else => error.BadRopeConfig,
    };
}

/// The `rope_parameters` key naming a RoPE other than default (YaRN for one), or null.
pub fn typed(rope: ?ObjectMap) ?[]const u8 {
    const r = rope orelse return null;
    for ([_][2][]const u8{ .{ "rope_type", "rope_parameters.rope_type" }, .{ "type", "rope_parameters.type" } }) |k| if (r.get(k[0])) |v| {
        if (v != .string or !std.mem.eql(u8, v.string, "default")) return k[1];
    };
    return null;
}

/// Whether the tower sets the legacy `rope_scaling`: transformers reads it as the config's `rope_parameters`.
pub fn legacy(tower: ObjectMap) bool {
    const v = tower.get("rope_scaling") orelse return false;
    return v != .null;
}

/// The key asking for scaled RoPE, in either form, or null for plain RoPE.
pub fn scaled(tower: ObjectMap, rope: ?ObjectMap) ?[]const u8 {
    if (typed(rope)) |key| return key;
    return if (legacy(tower)) "rope_scaling" else null;
}

/// Where `pick` looks: `rope_parameters` only, then the tower's top-level key, or the top-level key only when truthy.
pub const From = enum { params, both, truthy };

/// `key` from `rope_parameters`, else (per `from`) the tower's top-level key, else null; the caller converts it.
pub fn pick(rope: ?ObjectMap, tower: ObjectMap, key: []const u8, from: From) ?std.json.Value {
    if (rope) |r| if (r.get(key)) |v| return v;
    const v = switch (from) {
        .params => return null,
        .both, .truthy => tower.get(key) orelse return null,
    };
    return if (from == .truthy and !truthy(v)) null else v;
}

/// The block a scaled RoPE's parameters come from: a legacy `rope_scaling` object, else `rope_parameters`.
pub const Block = struct { map: ObjectMap, legacy: bool };

/// `rope_scaling` when it is an object, else `rope`; a `rope_scaling` neither null nor an object is refused.
pub fn block(tower: ObjectMap, rope: ObjectMap) Error!Block {
    const v = tower.get("rope_scaling") orelse return .{ .map = rope, .legacy = false };
    return switch (v) {
        .null => .{ .map = rope, .legacy = false },
        .object => |o| .{ .map = o, .legacy = true },
        else => error.BadRopeConfig,
    };
}

/// Python's truth of a JSON value.
fn truthy(v: std.json.Value) bool {
    return switch (v) {
        .null => false,
        .bool => |b| b,
        .integer => |i| i != 0,
        .float => |f| f != 0,
        .number_string, .string => |s| s.len != 0,
        .array => |l| l.items.len != 0,
        .object => |o| o.count() != 0,
    };
}

fn parsed(a: std.mem.Allocator, text: []const u8) !ObjectMap {
    return (try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{})).object;
}

test "rope config: YaRN in either form is named by its key; default, absent and null are plain" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { []const u8, ?[]const u8 }{
        .{ "{}", null },
        .{ "{\"rope_parameters\":null,\"rope_scaling\":null}", null },
        .{ "{\"rope_parameters\":{\"rope_type\":\"default\",\"rope_theta\":1e7}}", null },
        .{ "{\"rope_parameters\":{\"type\":\"default\"}}", null },
        .{ "{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":4}}", "rope_parameters.rope_type" },
        .{ "{\"rope_parameters\":{\"type\":\"yarn\"}}", "rope_parameters.type" },
        .{ "{\"rope_parameters\":{\"rope_type\":4}}", "rope_parameters.rope_type" },
        .{ "{\"rope_parameters\":{\"rope_type\":\"default\",\"type\":\"yarn\"}}", "rope_parameters.type" },
        .{ "{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":4}}", "rope_scaling" },
        .{ "{\"rope_scaling\":{\"type\":\"default\"}}", "rope_scaling" },
        .{ "{\"rope_parameters\":{\"rope_type\":\"yarn\"},\"rope_scaling\":{\"type\":\"yarn\"}}", "rope_parameters.rope_type" },
    };
    for (cases) |c| {
        const t = try parsed(a, c[0]);
        const got = scaled(t, try params(t));
        if (c[1]) |want| try std.testing.expectEqualStrings(want, got.?) else try std.testing.expectEqual(@as(?[]const u8, null), got);
    }
    try std.testing.expectError(error.BadRopeConfig, params(try parsed(a, "{\"rope_parameters\":4}")));
}

test "rope config: rope_parameters' key wins; the top-level key is read per From" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const both = try parsed(a, "{\"rope_theta\":5,\"rope_parameters\":{\"rope_theta\":7}}");
    for (std.enums.values(From)) |f| try std.testing.expectEqual(@as(i64, 7), pick(try params(both), both, "rope_theta", f).?.integer);
    const top = try parsed(a, "{\"rope_theta\":5,\"rope_parameters\":{}}");
    try std.testing.expectEqual(@as(?std.json.Value, null), pick(try params(top), top, "rope_theta", .params));
    try std.testing.expectEqual(@as(i64, 5), pick(try params(top), top, "rope_theta", .both).?.integer);
    try std.testing.expectEqual(@as(i64, 5), pick(try params(top), top, "rope_theta", .truthy).?.integer);
    const zero = try parsed(a, "{\"rope_theta\":0}");
    try std.testing.expectEqual(@as(i64, 0), pick(null, zero, "rope_theta", .both).?.integer);
    try std.testing.expectEqual(@as(?std.json.Value, null), pick(null, zero, "rope_theta", .truthy));
    const held = try parsed(a, "{\"rope_parameters\":{\"rope_theta\":null}}");
    try std.testing.expect(pick(try params(held), held, "rope_theta", .truthy).? == .null);
}

test "rope config: a legacy rope_scaling object is the block; null keeps rope_parameters; other values are refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const rope = try parsed(a, "{\"rope_type\":\"default\"}");
    const old = try block(try parsed(a, "{\"rope_scaling\":{\"type\":\"yarn\"}}"), rope);
    try std.testing.expect(old.legacy and old.map.get("type") != null);
    const none = try block(try parsed(a, "{\"rope_scaling\":null}"), rope);
    try std.testing.expect(!none.legacy and none.map.get("rope_type") != null);
    try std.testing.expect(!(try block(try parsed(a, "{}"), rope)).legacy);
    try std.testing.expectError(error.BadRopeConfig, block(try parsed(a, "{\"rope_scaling\":\"yarn\"}"), rope));
}
