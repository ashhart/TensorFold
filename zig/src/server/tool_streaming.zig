//! The streaming half of tool calls: each finished call's opening delta, then its arguments (``stream_tool_call_deltas``).
const std = @import("std");
const json = @import("json.zig");
const tool_specs = @import("tool_specs.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// ``stream_tool_call_deltas``: each call's opening delta, then its arguments in one more.
pub fn deltas(a: Allocator, calls: []const Value) Allocator.Error![]Value {
    var out: std.ArrayList(Value) = .empty;
    for (calls, 0..) |call, index| {
        const function = call.get("function") orelse continue;
        if (function != .object) continue;
        const head_fn = try json.newObject(a);
        try head_fn.put(a, "name", .{ .string = if (function.get("name")) |n| (if (n.truthy()) try tool_specs.pyStr(a, n) else "") else "" });
        try head_fn.put(a, "arguments", .{ .string = "" });
        const head = try json.newObject(a);
        try head.put(a, "index", try json.intValue(a, index));
        const id = call.get("id");
        try head.put(a, "id", .{ .string = if (id != null and id.?.truthy()) try tool_specs.pyStr(a, id.?) else try std.fmt.allocPrint(a, "call_{d}", .{index}) });
        const kind = call.get("type");
        try head.put(a, "type", .{ .string = if (kind != null and kind.?.truthy()) try tool_specs.pyStr(a, kind.?) else "function" });
        try head.put(a, "function", .{ .object = head_fn });
        try out.append(a, try wrapCalls(a, .{ .object = head }));
        const args = function.get("arguments");
        const text = if (args != null and args.?.truthy()) try tool_specs.pyStr(a, args.?) else "";
        if (text.len > 0) {
            const arg_fn = try json.newObject(a);
            try arg_fn.put(a, "arguments", .{ .string = text });
            const more = try json.newObject(a);
            try more.put(a, "index", try json.intValue(a, index));
            try more.put(a, "function", .{ .object = arg_fn });
            try out.append(a, try wrapCalls(a, .{ .object = more }));
        }
    }
    return out.items;
}

/// ``{"tool_calls": [one]}``: one streamed call delta.
pub fn wrapCalls(a: Allocator, one: Value) Allocator.Error!Value {
    const list = try a.alloc(Value, 1);
    list[0] = one;
    const delta = try json.newObject(a);
    try delta.put(a, "tool_calls", .{ .array = list });
    return .{ .object = delta };
}
