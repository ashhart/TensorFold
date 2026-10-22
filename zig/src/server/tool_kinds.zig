//! The JSON kinds a tool parameter's schema accepts, with its references and combiners resolved.
const std = @import("std");
const json = @import("json");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// The JSON kinds a schema can name.
pub const Kind = enum { array, object, boolean, integer, number, null, string };

/// The kinds a schema accepts.
pub const Kinds = std.EnumSet(Kind);

/// The kind a JSON value is.
fn kindOfValue(v: Value) Kind {
    return switch (v) {
        .array => .array,
        .object => .object,
        .bool => .boolean,
        .int => .integer,
        .float => .number,
        .null => .null,
        .string => .string,
    };
}

/// The kind in the set, with a number naming the integers as well.
fn insertKind(kinds: *Kinds, kind: Kind) void {
    kinds.insert(kind);
    if (kind == .number) kinds.insert(.integer);
}

/// How deep a chain of references may go before the schema is left alone.
const max_ref_depth = 32;

/// What a reference resolved to: still being resolved, or the kinds it names.
const Resolution = union(enum) {
    /// The reference is being resolved, so reaching it again is a cycle.
    in_progress,
    /// The kinds the reference names, or null when it names none.
    done: ?Kinds,
};

/// The kinds a schema accepts, with the references it points into resolved.
const Resolver = struct {
    a: Allocator,
    root: Value,
    /// What each reference resolved to, and that it is still being resolved.
    memo: std.StringHashMapUnmanaged(Resolution) = .empty,
    /// The references resolved so far.
    resolved: usize = 0,
    depth: usize = 0,

    /// The kinds the schema accepts: its own kinds and its combiners' kinds.
    fn kinds(self: *Resolver, schema: Value) Allocator.Error!?Kinds {
        const own = try self.ownKinds(schema);
        const combined = try self.combinedKinds(schema);
        const named = intersectKinds(own, combined) orelse return null;
        var result = named;
        if (result.contains(.number)) result.insert(.integer);
        return result;
    }

    /// The kinds the schema names itself: its type, its enum or const, and what its reference points at.
    fn ownKinds(self: *Resolver, schema: Value) Allocator.Error!?Kinds {
        if (schema != .object) return null;
        var named: ?Kinds = null;
        named = intersectKinds(named, typeKinds(schema.get("type")));
        if (schema.get("enum")) |e| {
            if (e == .array) named = intersectKinds(named, valueKinds(e.array));
        }
        if (schema.get("const")) |c| named = intersectKinds(named, valueKinds(&.{c}));
        if (schema.get("$ref")) |r| {
            if (r == .string) named = intersectKinds(named, try self.refKinds(r.string));
        }
        return named;
    }

    /// The kinds the schema's combiners name: oneOf, anyOf and allOf are constraints of their own.
    fn combinedKinds(self: *Resolver, schema: Value) Allocator.Error!?Kinds {
        if (schema != .object) return null;
        var named: ?Kinds = null;
        for ([_][]const u8{ "oneOf", "anyOf" }) |key| {
            const branches = schema.get(key) orelse continue;
            if (branches != .array) continue;
            var either: ?Kinds = null;
            for (branches.array) |branch| {
                const branch_kinds = (try self.kinds(branch)) orelse continue;
                if (branch_kinds.count() == 0) continue;
                either = if (either) |e| e.unionWith(branch_kinds) else branch_kinds;
            }
            named = intersectKinds(named, either);
        }
        if (schema.get("allOf")) |branches| {
            if (branches == .array) {
                var all: ?Kinds = null;
                for (branches.array) |branch| {
                    const branch_kinds = (try self.kinds(branch)) orelse continue;
                    if (branch_kinds.count() == 0) continue;
                    all = if (all) |a| a.intersectWith(branch_kinds) else branch_kinds;
                }
                named = intersectKinds(named, all);
            }
        }
        return named;
    }

    /// The kinds the reference points at, or null when it names none.
    fn refKinds(self: *Resolver, ref: []const u8) Allocator.Error!?Kinds {
        if (self.depth >= max_ref_depth) return null;
        if (self.memo.getEntry(ref)) |entry| switch (entry.value_ptr.*) {
            .in_progress => return null,
            .done => |found| return found,
        };
        const target = (try resolvePointer(self.a, self.root, ref)) orelse {
            try self.memo.put(self.a, ref, .{ .done = null });
            return null;
        };
        try self.memo.put(self.a, ref, .in_progress);
        self.resolved += 1;
        self.depth += 1;
        defer self.depth -= 1;
        const found = try self.kinds(target);
        try self.memo.put(self.a, ref, .{ .done = found });
        return found;
    }
};

/// The kinds a schema accepts, with every reference it points into resolved once.
pub fn schemaKinds(a: Allocator, value: Value, root: Value) Allocator.Error!?Kinds {
    var resolver: Resolver = .{ .a = a, .root = root };
    return resolver.kinds(value);
}

/// The kinds a schema's type names: one name, or a list of them.
fn typeKinds(t: ?Value) ?Kinds {
    const v = t orelse return null;
    var kinds: Kinds = .empty;
    switch (v) {
        .string => if (std.meta.stringToEnum(Kind, v.string)) |k| insertKind(&kinds, k),
        .array => for (v.array) |one| if (one == .string) if (std.meta.stringToEnum(Kind, one.string)) |k| insertKind(&kinds, k),
        else => {},
    }
    return if (kinds.count() > 0) kinds else null;
}

/// The kinds the values of an enum or a const are.
fn valueKinds(values: []const Value) ?Kinds {
    var kinds: Kinds = .empty;
    for (values) |v| insertKind(&kinds, kindOfValue(v));
    return if (kinds.count() > 0) kinds else null;
}

/// The kinds two sets share, with a set that names none constraining nothing.
fn intersectKinds(kinds: ?Kinds, other: ?Kinds) ?Kinds {
    const o = other orelse return kinds;
    const k = kinds orelse return o;
    return k.intersectWith(o);
}

/// The value a local reference points at, or null when it points nowhere.
fn resolvePointer(a: Allocator, root: Value, ref: []const u8) Allocator.Error!?Value {
    if (ref.len == 0 or ref[0] != '#') return null;
    var current = root;
    const fragment = (try percentDecode(a, ref[1..])) orelse return null;
    if (fragment.len == 0) return current;
    if (fragment[0] != '/') return null;
    var tokens = std.mem.splitScalar(u8, fragment[1..], '/');
    while (tokens.next()) |token| {
        const name = (try unescapePointer(a, token)) orelse return null;
        switch (current) {
            .object => current = current.object.get(name) orelse return null,
            .array => {
                const index = std.fmt.parseInt(usize, name, 10) catch return null;
                if (index >= current.array.len) return null;
                current = current.array[index];
            },
            else => return null,
        }
    }
    return current;
}

/// The fragment with its percent escapes decoded, or null when one is not a hex pair.
fn percentDecode(a: Allocator, text: []const u8) Allocator.Error!?[]const u8 {
    if (std.mem.indexOfScalar(u8, text, '%') == null) return text;
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if (text[i] != '%') {
            try out.append(a, text[i]);
            continue;
        }
        if (i + 2 >= text.len) return null;
        const hi = std.fmt.charToDigit(text[i + 1], 16) catch return null;
        const lo = std.fmt.charToDigit(text[i + 2], 16) catch return null;
        try out.append(a, hi * 16 + lo);
        i += 2;
    }
    return out.items;
}

/// The JSON pointer token with its `~1` and `~0` unescaped, or null when a tilde escapes nothing.
fn unescapePointer(a: Allocator, token: []const u8) Allocator.Error!?[]const u8 {
    if (std.mem.indexOfScalar(u8, token, '~') == null) return token;
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < token.len) : (i += 1) {
        if (token[i] != '~') {
            try out.append(a, token[i]);
            continue;
        }
        if (i + 1 >= token.len) return null;
        i += 1;
        try out.append(a, switch (token[i]) {
            '1' => '/',
            '0' => '~',
            else => return null,
        });
    }
    return out.items;
}

/// The schema takes a value that is not a string.
pub fn takesJson(kinds: Kinds) bool {
    var without_string = kinds;
    without_string.remove(.string);
    return without_string.count() > 0;
}

/// The parsed value is one the schema accepts; a string keeps the text the model wrote.
pub fn accepts(kinds: Kinds, v: Value) bool {
    const kind = kindOfValue(v);
    if (kind == .string) return false;
    return kinds.contains(kind);
}

test "a chain of definitions is resolved once each, not once per path" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const depth = 32;
    var defs: std.ArrayList(u8) = .empty;
    try defs.appendSlice(a, "\"$defs\":{");
    for (0..depth + 1) |i| {
        if (i > 0) try defs.appendSlice(a, ",");
        try defs.print(a, "\"D{d}\":{{\"anyOf\":[{{\"$ref\":\"#/$defs/D{d}\"}},{{\"$ref\":\"#/$defs/D{d}\"}}]}}", .{ i, i + 1, i + 1 });
    }
    try defs.appendSlice(a, "}");
    const root = (try json.parse(a, try std.mem.concat(a, u8, &.{ "{", defs.items, "}" }))).ok;
    const schema = (try json.parse(a, "{\"$ref\":\"#/$defs/D0\"}")).ok;
    var resolver: Resolver = .{ .a = a, .root = root };
    _ = try resolver.kinds(schema);
    // each definition is resolved once, so the work grows with the definitions and not with the paths
    try std.testing.expect(resolver.resolved <= depth + 1);
    try std.testing.expect(resolver.resolved > 1);
}
