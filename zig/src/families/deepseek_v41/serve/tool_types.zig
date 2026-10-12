//! Tool-call parameter values typed by their JSON schema, as the Python server reads them.
const std = @import("std");
const json = @import("json");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const tool_scan = @import("tool_scan.zig");
const pyStrip = tool_scan.pyStrip;

// -- values ----------------------------------------------------------------------------------------------------------

const TypeSet = packed struct(u8) { string: bool = false, integer: bool = false, number: bool = false, boolean: bool = false, array: bool = false, object: bool = false, null: bool = false, _: u1 = 0 };

fn typeOfName(name: []const u8) TypeSet {
    var t: TypeSet = .{};
    const eq = std.mem.eql;
    if (eq(u8, name, "string")) t.string = true;
    if (eq(u8, name, "integer")) t.integer = true;
    if (eq(u8, name, "number")) t.number = true;
    if (eq(u8, name, "boolean")) t.boolean = true;
    if (eq(u8, name, "array")) t.array = true;
    if (eq(u8, name, "object")) t.object = true;
    if (eq(u8, name, "null")) t.null = true;
    return t;
}

fn typeOfValue(v: Value) TypeSet {
    return switch (v) {
        .bool => .{ .boolean = true },
        .int => .{ .integer = true },
        .float => .{ .number = true },
        .string => .{ .string = true },
        .array => .{ .array = true },
        .object => .{ .object = true },
        .null => .{ .null = true },
    };
}

fn merge(x: TypeSet, y: TypeSet) TypeSet {
    return @bitCast(@as(u8, @bitCast(x)) | @as(u8, @bitCast(y)));
}

fn any(x: TypeSet) bool {
    return @as(u8, @bitCast(x)) != 0;
}

/// The JSON types a parameter schema allows, or null when silent; unknown type names give an empty set.
pub fn schemaTypes(prop: ?Value) ?TypeSet {
    const p = prop orelse return null;
    if (p != .object) return null;
    if (p.get("type")) |kind| {
        if (kind == .string) return typeOfName(kind.string);
        if (kind == .array and kind.array.len > 0) {
            var all = true;
            for (kind.array) |k| all = all and k == .string;
            if (all) {
                var t: TypeSet = .{};
                for (kind.array) |k| t = merge(t, typeOfName(k.string));
                return t;
            }
        }
    }
    for ([_][]const u8{ "anyOf", "oneOf" }) |key| if (p.get(key)) |members| if (members == .array and members.array.len > 0) {
        var t: TypeSet = .{};
        for (members.array) |m| t = merge(t, schemaTypes(m) orelse return null);
        return t;
    };
    if (p.get("enum")) |e| if (e == .array and e.array.len > 0) {
        var t: TypeSet = .{};
        for (e.array) |v| t = merge(t, typeOfValue(v));
        return t;
    };
    if (p.get("const")) |c| return typeOfValue(c);
    return null;
}

fn fits(v: Value, t: TypeSet) bool {
    const k = typeOfValue(v);
    return any(.{ .string = k.string and t.string, .integer = k.integer and t.integer, .number = k.number and t.number, .boolean = k.boolean and t.boolean, .array = k.array and t.array, .object = k.object and t.object, .null = k.null and t.null }) or (k.integer and t.number);
}

/// ``text`` with the arrays and objects it left open closed, or null if it does not only stop short.
pub fn closedJson(a: Allocator, text: []const u8) Allocator.Error!?[]u8 {
    var closers: std.ArrayList(u8) = .empty;
    var in_string = false;
    var escaped = false;
    for (text) |c| {
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else {
                escaped = c == '\\';
                in_string = c != '"';
            }
        } else if (c == '"') {
            in_string = true;
        } else if (c == '[' or c == '{') {
            try closers.append(a, if (c == '[') ']' else '}');
        } else if (c == ']' or c == '}') {
            if (closers.items.len == 0 or closers.pop().? != c) return null;
        }
    }
    if (closers.items.len == 0 or in_string) return null;
    std.mem.reverse(u8, closers.items);
    return try std.mem.concat(a, u8, &.{ pyRstrip(text), closers.items });
}

fn pyRstrip(s: []const u8) []const u8 {
    const stripped = pyStrip(s);
    return s[0 .. @intFromPtr(stripped.ptr) - @intFromPtr(s.ptr) + stripped.len];
}

fn loads(a: Allocator, text: []const u8) Allocator.Error!?Value {
    return switch (try json.parseText(a, text)) {
        .ok => |v| v,
        .err => null,
    };
}

/// An integral float as an integer; null past i128 (kept a float).
fn integral(a: Allocator, f: f64) Allocator.Error!?Value {
    if (@abs(f) >= 1.7e38) return null;
    return .{ .int = try std.fmt.allocPrint(a, "{d}", .{@as(i128, @intFromFloat(f))}) };
}

/// A value read by its parameter schema (see the module comment).
pub fn typedValue(a: Allocator, raw: []const u8, prop: ?Value) Allocator.Error!Value {
    const types = schemaTypes(prop) orelse {
        if (try loads(a, raw)) |v| return v;
        return .{ .string = raw };
    };
    if (types.string) return if (types.null and std.mem.eql(u8, pyStrip(raw), "null")) .null else .{ .string = raw };
    const text = pyStrip(raw);
    var candidates: std.ArrayList(Value) = .empty;
    if (try loads(a, text)) |v| try candidates.append(a, v);
    if (types.array or types.object) if (try closedJson(a, text)) |fixed| if (try loads(a, fixed)) |v| try candidates.append(a, v);
    if (candidates.items.len > 0 and candidates.items[0] == .string) if (try loads(a, pyStrip(candidates.items[0].string))) |v| try candidates.append(a, v);
    if (std.mem.eql(u8, text, "True")) try candidates.append(a, .{ .bool = true });
    if (std.mem.eql(u8, text, "False")) try candidates.append(a, .{ .bool = false });
    if (std.mem.eql(u8, text, "None")) try candidates.append(a, .null);
    for (candidates.items) |v| {
        var x = v;
        if (x == .float and types.integer and !types.number and std.math.isFinite(x.float) and @floor(x.float) == x.float) {
            if (try integral(a, x.float)) |n| x = n;
        }
        if (fits(x, types)) return x;
    }
    return .{ .string = raw };
}

/// A parameter's value from its text, ``string`` flag and schema.
pub fn paramValue(a: Allocator, raw: []const u8, string: bool, prop: ?Value) Allocator.Error!Value {
    const types = schemaTypes(prop);
    if (string) {
        if (types == null or types.?.string) return .{ .string = raw };
        return typedValue(a, raw, prop);
    }
    if (try loads(a, raw)) |v| return v;
    if (try closedJson(a, pyStrip(raw))) |fixed| if (try loads(a, fixed)) |v| return v;
    return if (types != null) typedValue(a, raw, prop) else .{ .string = raw };
}

/// ``{`` (the first) or ``,``, the key, ``:``, the compact value.
pub fn fragment(a: Allocator, out: *std.ArrayList(u8), first: bool, key: []const u8, v: Value) Allocator.Error!void {
    try out.append(a, if (first) '{' else ',');
    var w: std.Io.Writer.Allocating = .fromArrayList(a, out);
    json.writeString(&w.writer, key, .{ .ascii = false }) catch return error.OutOfMemory;
    w.writer.writeByte(':') catch return error.OutOfMemory;
    json.write(&w.writer, v, .{ .ascii = false, .compact = true }) catch return error.OutOfMemory;
    out.* = w.toArrayList();
}
