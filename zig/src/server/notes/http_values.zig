//! Notes HTTP values independent of inference, lookup and index construction.
const std = @import("std");
const json = @import("json");
const ops = @import("operations.zig");
const write = @import("write.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

fn stateName(s: @import("types.zig").State) []const u8 {
    return switch (s) {
        .current => "current",
        .superseded => "superseded",
        .tombstoned => "tombstoned",
    };
}

fn ids(a: Allocator, xs: []const u32) Allocator.Error!Value {
    const out = try a.alloc(Value, xs.len);
    for (xs, out) |x, *slot| slot.* = try json.intValue(a, x);
    return .{ .array = out };
}

pub fn viewValue(a: Allocator, v: ops.View) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "id", try json.intValue(a, v.id));
    try o.put(a, "version", try json.intValue(a, v.version));
    try o.put(a, "state", .{ .string = stateName(v.state) });
    try o.put(a, "kind", .{ .string = if (v.kind == .fact) "fact" else "take_back" });
    try o.put(a, "subject", .{ .string = v.subject });
    try o.put(a, "text", .{ .string = v.text });
    try o.put(a, "line", .{ .string = v.line });
    try o.put(a, "source", if (v.source) |s| .{ .string = s } else .null);
    try o.put(a, "told", try json.intValue(a, v.told));
    try o.put(a, "superseded_by", if (v.state == .superseded) try json.intValue(a, v.by) else .null);
    try o.put(a, "tombstoned_by", if (v.state == .tombstoned) try json.intValue(a, v.by) else .null);
    return .{ .object = o };
}

pub fn writtenValue(a: Allocator, op: write.Op, w: ops.Written) Allocator.Error!Value {
    const o = try json.newObject(a);
    if (w.note) |n| {
        try o.put(a, "id", try json.intValue(a, n.id));
        try o.put(a, "version", try json.intValue(a, n.version));
        try o.put(a, "state", .{ .string = stateName(n.state) });
    } else {
        try o.put(a, "id", .null);
        try o.put(a, "version", .null);
        try o.put(a, "state", .null);
    }
    try o.put(a, "verdict", .{ .string = @tagName(w.verdict) });
    try o.put(a, "written", .{ .bool = w.verdict.learn() });
    try o.put(a, "source", .{ .string = w.source });
    try o.put(a, "told", try json.intValue(a, w.told));
    try o.put(a, if (op == .take_back) "tombstoned" else "superseded", try ids(a, w.retired));
    try o.put(a, "same_as", try ids(a, w.same_as));
    try o.put(a, "note", if (w.note) |n| try viewValue(a, n) else .null);
    return .{ .object = o };
}
