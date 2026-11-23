const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const versions = [_]u32{1};
pub const Cursor = struct { count: usize = 0, last: ?u32 = null };

pub fn requested(cx: *errors.Cx, body: Value, streaming: bool, cortheon: bool) errors.Refused!bool {
    const options = body.get("stream_options") orelse return false;
    const version = options.get("tensorfold_token_stream_version") orelse return false;
    if (version != .int or version.int64() != 1)
        return cx.refuse("stream_options.tensorfold_token_stream_version must be the integer 1");
    if (!streaming or body.get("stream").? != .bool) return cx.refuse("tensorfold token streams require stream: true");
    if (cortheon) return cx.refuse("tensorfold token stream version 1 does not support Cortheon's multi-round replies");
    return true;
}

/// The chat sink's token callback for a stream sink type whose `run` is the reply it streams.
pub fn sinkTokens(comptime S: type) *const fn (*anyopaque, usize, []const u32) error{Closed}!void {
    return &struct {
        fn call(ctx: *anyopaque, offset: usize, ids: []const u32) error{Closed}!void {
            const s: *S = @ptrCast(@alignCast(ctx));
            return emit(s.run, offset, ids);
        }
    }.call;
}

pub fn emit(r: anytype, offset: usize, ids: []const u32) error{Closed}!void {
    if (ids.len == 0 or offset != r.token_cursor.count) return error.Closed;
    const value = payload(r.a, r.id, r.plan.named, offset, ids) catch return error.Closed;
    try r.out.vt.event(r.out.ctx, value);
    r.token_cursor = .{ .count = offset + ids.len, .last = ids[ids.len - 1] };
}

fn payload(a: Allocator, id: []const u8, model: []const u8, offset: usize, ids: []const u32) Allocator.Error!Value {
    const out = try json.newObject(a);
    try out.put(a, "object", .{ .string = "tensorfold.token.chunk" });
    try out.put(a, "version", .{ .int = "1" });
    try out.put(a, "id", .{ .string = id });
    try out.put(a, "model", .{ .string = model });
    try out.put(a, "offset", try json.intValue(a, offset));
    try out.put(a, "token_ids", try tokenIds(a, ids));
    return .{ .object = out };
}

pub fn tokenIds(a: Allocator, ids: []const u32) Allocator.Error!Value {
    const values = try a.alloc(Value, ids.len);
    for (values, ids) |*value, id| value.* = try json.intValue(a, id);
    return .{ .array = values };
}

pub fn anchor(a: Allocator, out: *json.Object, cursor: Cursor, delta: ?Value) Allocator.Error!void {
    const value = try json.newObject(a);
    try value.put(a, "version", .{ .int = "1" });
    try value.put(a, "token_count", try json.intValue(a, cursor.count));
    try value.put(a, "token_id", if (cursor.last) |id| try json.intValue(a, id) else .null);
    // each logprob entry's raw token position, in the shape of the chunk's logprobs
    if (delta) |d| if (d == .object) if (d.object.fetchOrderedRemove("logprob_positions")) |at|
        try value.put(a, "logprob_positions", at.value);
    try out.put(a, "tensorfold_stream", .{ .object = value });
}
