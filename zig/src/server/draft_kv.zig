//! Draft attention cache capacity, shared by the server's JSON reports.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");

pub fn put(a: std.mem.Allocator, o: *json.Object, cache: ?api.DraftKv) !void {
    const kv = cache orelse return;
    const value = try json.newObject(a);
    try value.put(a, "bytes", try json.intValue(a, kv.bytes));
    try value.put(a, "dtype", .{ .string = @tagName(kv.dtype) });
    try value.put(a, "scope", .{ .string = @tagName(kv.scope) });
    try o.put(a, "draft_kv", .{ .object = value });
}
