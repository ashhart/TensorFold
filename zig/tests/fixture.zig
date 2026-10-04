//! Lane test fixtures from TF_LANES_FIXTURES: JSON lines, floats as IEEE bits, u64 past i64 as number strings.
const std = @import("std");
const lanes = @import("lanes");
const json = std.json;
const Allocator = std.mem.Allocator;

pub const Value = json.Value;

/// The fixtures directory, or null when the environment does not name one (the tests then skip).
pub fn dir() ?[]const u8 {
    return std.testing.environ.getPosix("TF_LANES_FIXTURES");
}

pub fn read(gpa: Allocator, name: []const u8) ![]u8 {
    const root = dir() orelse return error.SkipZigTest;
    const path = try std.fs.path.join(gpa, &.{ root, name });
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, gpa, .limited(1 << 30)) catch |err| switch (err) {
        error.FileNotFound => error.SkipZigTest,
        else => err,
    };
}

/// Every line of a JSONL file, parsed (into `arena`), with its text.
pub const Line = struct { text: []const u8, value: Value };

pub fn lines(arena: Allocator, text: []const u8) ![]Line {
    var out: std.ArrayList(Line) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |t| {
        if (t.len == 0) continue;
        try out.append(arena, .{ .text = t, .value = try json.parseFromSliceLeaky(Value, arena, t, .{}) });
    }
    return out.items;
}

pub fn get(v: Value, key: []const u8) Value {
    return v.object.get(key) orelse std.debug.panic("fixture field {s} missing", .{key});
}

pub fn isNull(v: Value) bool {
    return v == .null;
}

pub fn int(v: Value) i64 {
    return switch (v) {
        .integer => |x| x,
        else => std.debug.panic("fixture: not an integer", .{}),
    };
}

pub fn uint(v: Value) u64 {
    return switch (v) {
        .integer => |x| @bitCast(x),
        .number_string => |s| std.fmt.parseInt(u64, s, 10) catch unreachable,
        else => std.debug.panic("fixture: not an unsigned integer", .{}),
    };
}

/// A float the recorder wrote as its bits.
pub fn float(v: Value) f64 {
    return @bitCast(uint(v));
}

pub fn boolean(v: Value) bool {
    return v.bool;
}

pub fn str(v: Value) []const u8 {
    return v.string;
}

pub fn u32s(a: Allocator, v: Value) ![]u32 {
    const out = try a.alloc(u32, v.array.items.len);
    for (out, v.array.items) |*o, x| o.* = @intCast(int(x));
    return out;
}

pub fn i64s(a: Allocator, v: Value) ![]i64 {
    const out = try a.alloc(i64, v.array.items.len);
    for (out, v.array.items) |*o, x| o.* = int(x);
    return out;
}

pub fn floats(a: Allocator, v: Value) ![]f64 {
    const out = try a.alloc(f64, v.array.items.len);
    for (out, v.array.items) |*o, x| o.* = float(x);
    return out;
}

/// [[key, bits], ...] pairs.
pub const Pair = struct { key: i64, value: f64 };

pub fn pairs(a: Allocator, v: Value) ![]Pair {
    const out = try a.alloc(Pair, v.array.items.len);
    for (out, v.array.items) |*o, x| o.* = .{ .key = int(x.array.items[0]), .value = float(x.array.items[1]) };
    return out;
}

pub fn sampling(v: Value) lanes.Sampling {
    return .{ .seed = uint(get(v, "seed")), .temperature = float(get(v, "temperature")), .top_k = @intCast(int(get(v, "top_k"))), .top_p = float(get(v, "top_p")), .min_p = float(get(v, "min_p")) };
}

pub fn expectBits(want: f64, got: f64) !void {
    try std.testing.expectEqual(@as(u64, @bitCast(want)), @as(u64, @bitCast(got)));
}

/// The recorded pairs are exactly the table's set keys, values bit for bit.
pub fn expectTable(a: Allocator, want: Value, got: lanes.table.Table) !void {
    const ps = try pairs(a, want);
    var set: usize = 0;
    for (got.slots) |slot| set += @intFromBool(slot != null);
    try std.testing.expectEqual(ps.len, set);
    for (ps) |p| try expectBits(p.value, got.get(p.key).?);
}
