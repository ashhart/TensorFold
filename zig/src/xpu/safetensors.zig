//! A safetensors file's header, checked against the file's own bytes before any tensor is sliced (host-tested).
const std = @import("std");

pub const Tensor = struct { name: []const u8, dtype: []const u8, shape: [8]u64, rank: usize, bytes: []const u8 };

pub const Error = error{ BadSafetensors, TensorRank, UnknownDtype } || std.mem.Allocator.Error;

fn dtypeBytes(dtype: []const u8) ?u64 {
    const sizes = [_]struct { []const u8, u64 }{
        .{ "BOOL", 1 }, .{ "U8", 1 },  .{ "I8", 1 },  .{ "F8_E4M3", 1 }, .{ "F8_E5M2", 1 },
        .{ "U16", 2 },  .{ "I16", 2 }, .{ "F16", 2 }, .{ "BF16", 2 },    .{ "U32", 4 },
        .{ "I32", 4 },  .{ "F32", 4 }, .{ "U64", 8 }, .{ "I64", 8 },     .{ "F64", 8 },
    };
    for (sizes) |s| if (std.mem.eql(u8, s[0], dtype)) return s[1];
    return null;
}

fn count(v: ?std.json.Value) Error!u64 {
    const x = v orelse return error.BadSafetensors;
    if (x != .integer or x.integer < 0) return error.BadSafetensors;
    return @intCast(x.integer);
}

/// Appends `file`'s tensors to `out`, offsets and shape × dtype checked against the file; names live in the result.
pub fn parse(gpa: std.mem.Allocator, file: []const u8, out: *std.ArrayList(Tensor)) Error!std.json.Parsed(std.json.Value) {
    if (file.len < 8) return error.BadSafetensors;
    const n = std.mem.readInt(u64, file[0..8], .little);
    if (n > file.len - 8) return error.BadSafetensors;
    const data = file[8 + n ..];
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, file[8 .. 8 + n], .{}) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.BadSafetensors,
    };
    errdefer parsed.deinit();
    const had = out.items.len;
    errdefer out.shrinkRetainingCapacity(had);
    if (parsed.value != .object) return error.BadSafetensors;
    var it = parsed.value.object.iterator();
    while (it.next()) |e| {
        if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
        if (e.value_ptr.* != .object) return error.BadSafetensors;
        const o = e.value_ptr.object;
        const dtype = o.get("dtype") orelse return error.BadSafetensors;
        const shape = o.get("shape") orelse return error.BadSafetensors;
        const span = o.get("data_offsets") orelse return error.BadSafetensors;
        if (dtype != .string or shape != .array or span != .array or span.array.items.len != 2) return error.BadSafetensors;
        var t: Tensor = .{ .name = e.key_ptr.*, .dtype = dtype.string, .shape = @splat(1), .rank = shape.array.items.len, .bytes = &.{} };
        if (t.rank > t.shape.len) return error.TensorRank; // vision conv is 5-D
        var bytes = dtypeBytes(t.dtype) orelse return error.UnknownDtype;
        for (shape.array.items, 0..) |d, i| {
            t.shape[i] = try count(d);
            bytes = std.math.mul(u64, bytes, t.shape[i]) catch return error.BadSafetensors;
        }
        const from = try count(span.array.items[0]);
        const to = try count(span.array.items[1]);
        if (from > to or to > data.len or to - from != bytes) return error.BadSafetensors;
        t.bytes = data[@intCast(from)..@intCast(to)];
        try out.append(gpa, t);
    }
    return parsed;
}

fn sample(comptime header: []const u8, comptime data_len: usize) [8 + header.len + data_len]u8 {
    var f: [8 + header.len + data_len]u8 = @splat(0);
    std.mem.writeInt(u64, f[0..8], header.len, .little);
    @memcpy(f[8 .. 8 + header.len], header);
    return f;
}

fn expectParse(f: []const u8) Error!usize {
    var out: std.ArrayList(Tensor) = .empty;
    defer out.deinit(std.testing.allocator);
    const p = try parse(std.testing.allocator, f, &out);
    p.deinit();
    return out.items.len;
}

test "a well-formed file gives its tensors" {
    const f = sample("{\"__metadata__\":{},\"a\":{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[0,12]}}", 12);
    try std.testing.expectEqual(@as(usize, 1), try expectParse(&f));
}

test "a header longer than the file is refused" {
    var f = sample("{}", 0);
    std.mem.writeInt(u64, f[0..8], 1 << 40, .little);
    try std.testing.expectError(error.BadSafetensors, expectParse(&f));
    try std.testing.expectError(error.BadSafetensors, expectParse(f[0..5]));
}

test "offsets past the data, reversed, or not matching the shape are refused" {
    const past = sample("{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}", 8);
    try std.testing.expectError(error.BadSafetensors, expectParse(&past));
    const reversed = sample("{\"a\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[8,0]}}", 8);
    try std.testing.expectError(error.BadSafetensors, expectParse(&reversed));
    const short = sample("{\"a\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,6]}}", 8);
    try std.testing.expectError(error.BadSafetensors, expectParse(&short));
    const negative = sample("{\"a\":{\"dtype\":\"U8\",\"shape\":[-1],\"data_offsets\":[0,0]}}", 0);
    try std.testing.expectError(error.BadSafetensors, expectParse(&negative));
}

test "a missing field or an unknown dtype is refused" {
    const no_shape = sample("{\"a\":{\"dtype\":\"U8\",\"data_offsets\":[0,1]}}", 1);
    try std.testing.expectError(error.BadSafetensors, expectParse(&no_shape));
    const unknown = sample("{\"a\":{\"dtype\":\"Q4\",\"shape\":[1],\"data_offsets\":[0,1]}}", 1);
    try std.testing.expectError(error.UnknownDtype, expectParse(&unknown));
}
